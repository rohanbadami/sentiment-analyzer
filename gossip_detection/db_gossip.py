"""
db_gossip.py - MySQL schema and CRUD operations for gossip detection.

Extends db_mysql.py with tables for reddit_events, ticker_mentions, and gossip_scores.
Uses SQLAlchemy engine and batch insert patterns from the main repo.
"""

import sys
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Import db_mysql utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_mysql import get_engine, executemany_update

from gossip_detection.config import (
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_DATABASE,
)

logger = logging.getLogger(__name__)


def ensure_gossip_tables():
    """
    Create gossip detection tables if they don't exist.

    Tables:
    - reddit_events: Raw Reddit posts/comments
    - ticker_mentions: Ticker extractions + bucketing
    - gossip_scores: Computed gossip scores per ticker
    """
    engine = get_engine()
    if not engine:
        logger.error("Failed to get database engine")
        return

    with engine.connect() as conn:
        try:
            # ================================================================
            # Table 1: reddit_events
            # ================================================================
            create_reddit_events = """
            CREATE TABLE IF NOT EXISTS reddit_events (
                id INT AUTO_INCREMENT PRIMARY KEY,
                reddit_id VARCHAR(32) UNIQUE NOT NULL,
                source VARCHAR(16) DEFAULT 'reddit',
                subreddit VARCHAR(64) NOT NULL,
                author VARCHAR(64) NOT NULL,
                author_karma INT,
                title LONGTEXT,
                text LONGTEXT NOT NULL,
                post_type VARCHAR(16) NOT NULL,
                created_utc DATETIME NOT NULL,
                upvotes INT DEFAULT 0,
                num_comments INT DEFAULT 0,
                permalink VARCHAR(512),
                engagement_score DOUBLE DEFAULT 0.0,
                raw_metadata JSON,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_subreddit (subreddit),
                INDEX idx_created_utc (created_utc),
                INDEX idx_author (author)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """
            conn.execute(text(create_reddit_events))
            logger.info("Ensured reddit_events table")

            # ================================================================
            # Table 2: ticker_mentions
            # ================================================================
            create_ticker_mentions = """
            CREATE TABLE IF NOT EXISTS ticker_mentions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                event_id INT NOT NULL,
                ticker VARCHAR(16) NOT NULL,
                source VARCHAR(16) DEFAULT 'reddit',
                mentioned_at DATETIME NOT NULL,
                bucket_5m DATETIME,
                bucket_15m DATETIME,
                bucket_60m DATETIME,
                FOREIGN KEY (event_id) REFERENCES reddit_events(id) ON DELETE CASCADE,
                INDEX idx_ticker (ticker),
                INDEX idx_bucket_5m (ticker, bucket_5m),
                INDEX idx_bucket_15m (ticker, bucket_15m),
                INDEX idx_bucket_60m (ticker, bucket_60m)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """
            conn.execute(text(create_ticker_mentions))
            logger.info("Ensured ticker_mentions table")

            # ================================================================
            # Table 3: gossip_scores
            # ================================================================
            create_gossip_scores = """
            CREATE TABLE IF NOT EXISTS gossip_scores (
                id INT AUTO_INCREMENT PRIMARY KEY,
                ticker VARCHAR(16) NOT NULL,
                computed_at DATETIME NOT NULL,
                gossip_score DOUBLE NOT NULL,
                mention_count INT DEFAULT 0,
                mention_velocity DOUBLE DEFAULT 0.0,
                acceleration DOUBLE DEFAULT 0.0,
                unique_authors INT DEFAULT 0,
                top_phrases JSON,
                velocity_metrics JSON,
                confirmation_flag BOOLEAN DEFAULT FALSE,
                alert_triggered BOOLEAN DEFAULT FALSE,
                INDEX idx_ticker_time (ticker, computed_at)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """
            conn.execute(text(create_gossip_scores))
            logger.info("Ensured gossip_scores table")

            conn.commit()
            logger.info("All gossip tables created successfully")

        except SQLAlchemyError as e:
            logger.error(f"Database error creating tables: {e}")
            conn.rollback()
        except Exception as e:
            logger.error(f"Unexpected error creating tables: {e}")


def bulk_insert_events(events: list[dict]) -> int:
    """
    Bulk insert Reddit events into reddit_events table.

    Two-phase insert: bulk first, then row-by-row for duplicates (following
    db_mysql.py pattern).

    Args:
        events: List of dicts with event data (from RedditEvent.to_dict())

    Returns:
        Number of successfully inserted events
    """
    if not events:
        return 0

    engine = get_engine()
    if not engine:
        logger.error("Failed to get database engine")
        return 0

    inserted_count = 0

    with engine.connect() as conn:
        # Bulk insert attempt
        try:
            insert_sql = """
            INSERT INTO reddit_events
            (reddit_id, source, subreddit, author, author_karma, title, text,
             post_type, created_utc, upvotes, num_comments, permalink, engagement_score, raw_metadata)
            VALUES (:reddit_id, :source, :subreddit, :author, :author_karma, :title, :text,
                    :post_type, :created_utc, :upvotes, :num_comments, :permalink, :engagement_score, :raw_metadata)
            """

            # Ensure datetime fields are ISO strings
            for event in events:
                if isinstance(event.get('created_utc'), datetime):
                    event['created_utc'] = event['created_utc'].isoformat()

            conn.execute(text(insert_sql), events)
            conn.commit()
            inserted_count = len(events)
            logger.info(f"Bulk inserted {inserted_count} reddit events")

        except Exception as e:
            logger.warning(f"Bulk insert failed: {e}. Falling back to row-by-row...")
            conn.rollback()

            # Fall back to row-by-row insert (skip duplicates)
            for event in events:
                try:
                    if isinstance(event.get('created_utc'), datetime):
                        event['created_utc'] = event['created_utc'].isoformat()

                    conn.execute(text(insert_sql), [event])
                    conn.commit()
                    inserted_count += 1
                except Exception as row_error:
                    if "Duplicate entry" in str(row_error):
                        logger.debug(f"Skipped duplicate: {event.get('reddit_id')}")
                    else:
                        logger.error(f"Row insert failed: {row_error}")
                    conn.rollback()

    return inserted_count


def bulk_insert_mentions(mentions: list[dict]) -> int:
    """
    Bulk insert ticker mentions into ticker_mentions table.

    Args:
        mentions: List of dicts with mention data

    Returns:
        Number of successfully inserted mentions
    """
    if not mentions:
        return 0

    engine = get_engine()
    if not engine:
        logger.error("Failed to get database engine")
        return 0

    inserted_count = 0

    with engine.connect() as conn:
        try:
            insert_sql = """
            INSERT INTO ticker_mentions
            (event_id, ticker, source, mentioned_at, bucket_5m, bucket_15m, bucket_60m)
            VALUES (:event_id, :ticker, :source, :mentioned_at, :bucket_5m, :bucket_15m, :bucket_60m)
            """

            # Convert datetime to ISO strings
            for mention in mentions:
                for field in ['mentioned_at', 'bucket_5m', 'bucket_15m', 'bucket_60m']:
                    if isinstance(mention.get(field), datetime):
                        mention[field] = mention[field].isoformat()

            conn.execute(text(insert_sql), mentions)
            conn.commit()
            inserted_count = len(mentions)
            logger.info(f"Bulk inserted {inserted_count} ticker mentions")

        except Exception as e:
            logger.warning(f"Bulk insert mentions failed: {e}. Falling back to row-by-row...")
            conn.rollback()

            for mention in mentions:
                try:
                    for field in ['mentioned_at', 'bucket_5m', 'bucket_15m', 'bucket_60m']:
                        if isinstance(mention.get(field), datetime):
                            mention[field] = mention[field].isoformat()

                    conn.execute(text(insert_sql), [mention])
                    conn.commit()
                    inserted_count += 1
                except Exception as row_error:
                    logger.debug(f"Row insert failed: {row_error}")
                    conn.rollback()

    return inserted_count


def insert_gossip_score(score: dict) -> bool:
    """
    Insert a single gossip score into gossip_scores table.

    Args:
        score: Dict with gossip score data

    Returns:
        True if insert successful, False otherwise
    """
    engine = get_engine()
    if not engine:
        logger.error("Failed to get database engine")
        return False

    try:
        insert_sql = """
        INSERT INTO gossip_scores
        (ticker, computed_at, gossip_score, mention_count, mention_velocity, acceleration,
         unique_authors, top_phrases, velocity_metrics, confirmation_flag, alert_triggered)
        VALUES (:ticker, :computed_at, :gossip_score, :mention_count, :mention_velocity, :acceleration,
                :unique_authors, :top_phrases, :velocity_metrics, :confirmation_flag, :alert_triggered)
        """

        # Convert datetime to ISO string
        if isinstance(score.get('computed_at'), datetime):
            score['computed_at'] = score['computed_at'].isoformat()

        with engine.connect() as conn:
            conn.execute(text(insert_sql), [score])
            conn.commit()
            return True

    except Exception as e:
        logger.error(f"Failed to insert gossip score: {e}")
        return False


def get_recent_mentions(ticker: str, minutes: int = 60) -> pd.DataFrame:
    """
    Get recent mentions for a ticker within a time window.

    Args:
        ticker: Stock ticker symbol
        minutes: Lookback window in minutes

    Returns:
        DataFrame with columns: [event_id, ticker, mentioned_at, upvotes, author]
    """
    engine = get_engine()
    if not engine:
        logger.error("Failed to get database engine")
        return pd.DataFrame()

    try:
        query = """
        SELECT
            tm.event_id,
            tm.ticker,
            tm.mentioned_at,
            re.upvotes,
            re.author
        FROM ticker_mentions tm
        JOIN reddit_events re ON tm.event_id = re.id
        WHERE tm.ticker = :ticker
        AND tm.mentioned_at >= DATE_SUB(NOW(), INTERVAL :minutes MINUTE)
        ORDER BY tm.mentioned_at DESC
        """

        with engine.connect() as conn:
            result = conn.execute(
                text(query),
                {"ticker": ticker, "minutes": minutes}
            )
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df

    except Exception as e:
        logger.error(f"Error querying recent mentions: {e}")
        return pd.DataFrame()


def get_latest_gossip_scores(limit: int = 50) -> pd.DataFrame:
    """
    Get the latest gossip scores for all tickers.

    Args:
        limit: Maximum number of rows to return

    Returns:
        DataFrame with latest gossip scores, sorted by score descending
    """
    engine = get_engine()
    if not engine:
        logger.error("Failed to get database engine")
        return pd.DataFrame()

    try:
        # Get latest computed_at for each ticker
        query = """
        SELECT
            ticker,
            computed_at,
            gossip_score,
            mention_count,
            mention_velocity,
            acceleration,
            unique_authors,
            top_phrases,
            confirmation_flag,
            alert_triggered
        FROM gossip_scores
        WHERE (ticker, computed_at) IN (
            SELECT ticker, MAX(computed_at)
            FROM gossip_scores
            GROUP BY ticker
        )
        ORDER BY gossip_score DESC
        LIMIT :limit
        """

        with engine.connect() as conn:
            result = conn.execute(text(query), {"limit": limit})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df

    except Exception as e:
        logger.error(f"Error querying gossip scores: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # Create tables when run directly
    ensure_gossip_tables()
    print("Gossip tables initialized")
