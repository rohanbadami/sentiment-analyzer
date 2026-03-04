"""
run_pipeline.py - Main orchestrator for the gossip detection pipeline.

Runs continuous polling cycles:
1. Fetch Reddit events
2. Extract tickers
3. Store in database
4. Compute gossip scores
5. Detect anomalies
6. Sleep and repeat

Graceful shutdown via signal handlers.
"""

import logging
import signal
import sys
import time
from datetime import datetime

from gossip_detection.config import POLL_INTERVAL_SECONDS, CHECKPOINT_FILE
from gossip_detection.reddit_ingestion import RedditPoller
from gossip_detection.ticker_extractor import TickerExtractor
from gossip_detection.db_gossip import ensure_gossip_tables, bulk_insert_events, bulk_insert_mentions
from gossip_detection.rolling_tracker import RollingTracker
from gossip_detection.gossip_scorer import GossipScorer
from gossip_detection.anomaly_detector import AnomalyDetector
from gossip_detection.redis_client import GossipRedisClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
SHUTDOWN_REQUESTED = False


def signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    global SHUTDOWN_REQUESTED
    logger.info(f"Received signal {signum}. Shutting down gracefully...")
    SHUTDOWN_REQUESTED = True


def run_cycle(
    poller: RedditPoller,
    extractor: TickerExtractor,
    redis_client: GossipRedisClient,
    scorer: GossipScorer,
):
    """
    Execute a single pipeline cycle.

    Steps:
    1. Poll Reddit
    2. Extract tickers from events
    3. Bulk insert events and mentions
    4. Store in Redis
    5. Compute gossip scores
    6. Store alerts

    Args:
        poller: RedditPoller instance
        extractor: TickerExtractor instance
        redis_client: GossipRedisClient instance
        scorer: GossipScorer instance
    """
    cycle_start = time.time()

    logger.info("=" * 70)
    logger.info("Starting pipeline cycle")

    # ========================================================================
    # Step 1: Poll Reddit
    # ========================================================================
    try:
        since_utc = int(poller.last_poll_utc)
        events = poller.poll_all(since_utc)
        logger.info(f"Fetched {len(events)} Reddit events")
    except Exception as e:
        logger.error(f"Poll failed: {e}")
        return

    if not events:
        logger.info("No events fetched this cycle")
        return

    # ========================================================================
    # Step 2: Extract tickers and prepare for DB insertion
    # ========================================================================
    event_dicts = []
    mention_dicts = []

    for event in events:
        # Extract tickers
        tickers = extractor.extract_tickers(event.text)
        event.tickers = tickers

        # Prepare event dict
        event_dict = event.to_dict()
        event_dicts.append(event_dict)

    logger.info(f"Extracted tickers from {len(events)} events")

    # ========================================================================
    # Step 3: Bulk insert events
    # ========================================================================
    try:
        inserted_count = bulk_insert_events(event_dicts)
        logger.info(f"Inserted {inserted_count} events to database")
    except Exception as e:
        logger.error(f"Failed to insert events: {e}")
        return

    # ========================================================================
    # Step 4: Prepare ticker mentions (with bucket timestamps)
    # ========================================================================
    try:
        # Query inserted event IDs for ticker mentions
        from db_mysql import get_engine
        from sqlalchemy import text

        engine = get_engine()

        # Batch query recently inserted events
        with engine.connect() as conn:
            query = """
            SELECT id, reddit_id, created_utc FROM reddit_events
            WHERE reddit_id IN ({})
            """

            reddit_ids = [ed["reddit_id"] for ed in event_dicts]
            if reddit_ids:
                # Build IN clause
                in_clause = ", ".join([f"'{rid}'" for rid in reddit_ids])
                query = query.format(in_clause)

                result = conn.execute(text(query))
                event_rows = result.fetchall()

                # Map reddit_id to event_id
                id_map = {row[1]: row[0] for row in event_rows}

                # Build mentions
                for event in events:
                    if not event.tickers:
                        continue

                    event_id = id_map.get(event.reddit_id)
                    if not event_id:
                        continue

                    created_utc = event.created_utc

                    # Compute bucket timestamps
                    utc_ts = int(created_utc.timestamp())
                    bucket_5m = (utc_ts // 300) * 300
                    bucket_15m = (utc_ts // 900) * 900
                    bucket_60m = (utc_ts // 3600) * 3600

                    from datetime import datetime as dt

                    for ticker in event.tickers:
                        mention_dict = {
                            "event_id": event_id,
                            "ticker": ticker,
                            "source": "reddit",
                            "mentioned_at": created_utc.isoformat(),
                            "bucket_5m": dt.utcfromtimestamp(bucket_5m).isoformat(),
                            "bucket_15m": dt.utcfromtimestamp(bucket_15m).isoformat(),
                            "bucket_60m": dt.utcfromtimestamp(bucket_60m).isoformat(),
                        }
                        mention_dicts.append(mention_dict)

        logger.info(f"Prepared {len(mention_dicts)} ticker mentions")

    except Exception as e:
        logger.error(f"Failed to prepare mentions: {e}")

    # ========================================================================
    # Step 5: Bulk insert ticker mentions
    # ========================================================================
    if mention_dicts:
        try:
            inserted_mentions = bulk_insert_mentions(mention_dicts)
            logger.info(f"Inserted {inserted_mentions} ticker mentions")
        except Exception as e:
            logger.error(f"Failed to insert mentions: {e}")

    # ========================================================================
    # Step 6: Store latest pull in Redis
    # ========================================================================
    try:
        redis_client.store_latest_pull(event_dicts, datetime.utcnow())
        logger.info("Stored latest pull in Redis")
    except Exception as e:
        logger.error(f"Failed to store in Redis: {e}")

    # ========================================================================
    # Step 7: Compute gossip scores
    # ========================================================================
    try:
        scores = scorer.score_all_active()
        if scores:
            logger.info(f"Computed scores for {len(scores)} tickers")
            top_ticker = scores[0]
            logger.info(
                f"  Top: {top_ticker['ticker']} (score={top_ticker['gossip_score']:.3f})"
            )
    except Exception as e:
        logger.error(f"Failed to compute scores: {e}")

    # ========================================================================
    # Cycle summary
    # ========================================================================
    cycle_end = time.time()
    cycle_time_ms = (cycle_end - cycle_start) * 1000

    logger.info(f"Cycle complete in {cycle_time_ms:.0f}ms")
    logger.info("=" * 70)


def main():
    """
    Main pipeline loop.

    Runs continuous polling cycles with graceful shutdown support.
    """
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Gossip Detection Pipeline Starting...")

    # ========================================================================
    # Initialize database and tables
    # ========================================================================
    try:
        ensure_gossip_tables()
        logger.info("Database tables initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return

    # ========================================================================
    # Initialize components
    # ========================================================================
    try:
        from gossip_detection.config import SUBREDDITS
        poller = RedditPoller(subreddits=SUBREDDITS)
        extractor = TickerExtractor()
        redis_client = GossipRedisClient()
        tracker = RollingTracker()
        anomaly_detector = AnomalyDetector()
        scorer = GossipScorer(
            tracker=tracker,
            anomaly_detector=anomaly_detector,
        )
        logger.info("All components initialized")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return

    # ========================================================================
    # Main loop
    # ========================================================================
    cycle_count = 0

    while not SHUTDOWN_REQUESTED:
        try:
            cycle_count += 1
            logger.info(f"\n--- Cycle {cycle_count} ---")

            run_cycle(poller, extractor, redis_client, scorer)

            # Sleep until next cycle
            logger.info(f"Sleeping for {POLL_INTERVAL_SECONDS} seconds...")
            time.sleep(POLL_INTERVAL_SECONDS)

        except Exception as e:
            logger.exception(f"Unhandled exception in main loop: {e}")
            # Don't crash; continue to next cycle
            time.sleep(POLL_INTERVAL_SECONDS)

    logger.info("Pipeline shutdown complete")


if __name__ == "__main__":
    main()
