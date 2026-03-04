"""
gossip_scorer.py - Computes composite gossip scores for tickers.

Ensemble approach combining 5 weighted signals:
1. Mention velocity (mentions/min)
2. Acceleration (velocity spike)
3. Author diversity (unique authors)
4. Anomaly detection (z-score spike)
5. Cross-source confirmation (Reddit x StockTwits)

Follows the pattern from integrated_processor.py (weighted voting, normalized [0,1]).
"""

import logging
from datetime import datetime
from typing import Optional

from gossip_detection.rolling_tracker import RollingTracker
from gossip_detection.anomaly_detector import AnomalyDetector
from gossip_detection.cross_source import CrossSourceConfirmer
from gossip_detection.phrase_engine import PhraseEngine
from gossip_detection.config import (
    GOSSIP_SCORE_WEIGHTS,
    VELOCITY_MAX_MENTIONS_PER_MIN,
    ACCELERATION_MAX,
    AUTHOR_DIVERSITY_MAX,
)
from gossip_detection.db_gossip import insert_gossip_score, get_recent_mentions
from gossip_detection.redis_client import GossipRedisClient

logger = logging.getLogger(__name__)


class GossipScorer:
    """
    Computes composite gossip score for each ticker.

    Combines 5 normalized sub-scores via weighted voting:
    - Velocity (25%): mentions per minute
    - Acceleration (25%): velocity increase
    - Author diversity (15%): unique authors
    - Anomaly score (20%): z-score spike
    - Confirmation (15%): cross-source phrase overlap
    """

    def __init__(
        self,
        tracker: Optional[RollingTracker] = None,
        anomaly_detector: Optional[AnomalyDetector] = None,
        confirmer: Optional[CrossSourceConfirmer] = None,
        phrase_engine: Optional[PhraseEngine] = None,
    ):
        """
        Initialize gossip scorer with dependencies.

        Args:
            tracker: RollingTracker instance (created if None)
            anomaly_detector: AnomalyDetector instance (created if None)
            confirmer: CrossSourceConfirmer instance (created if None)
            phrase_engine: PhraseEngine instance (created if None)
        """
        self.tracker = tracker or RollingTracker()
        self.anomaly_detector = anomaly_detector or AnomalyDetector()
        self.confirmer = confirmer or CrossSourceConfirmer()
        self.phrase_engine = phrase_engine or PhraseEngine()
        self.redis_client = GossipRedisClient()

        # Verify weights sum to 1.0
        total_weight = sum(GOSSIP_SCORE_WEIGHTS.values())
        assert abs(total_weight - 1.0) < 0.001, f"Weights must sum to 1.0, got {total_weight}"

    def compute_score(self, ticker: str) -> dict:
        """
        Compute composite gossip score for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict with gossip_score and component metrics
        """
        # Get 15m metrics (primary signal window)
        metrics = self.tracker.compute_all_metrics(ticker)
        metrics_15m = metrics["windows"]["15m"]

        mention_count = metrics_15m["count"]
        velocity_15m = metrics_15m["velocity"]
        acceleration_15m = metrics_15m["acceleration"]
        unique_authors = metrics_15m["unique_authors"]

        # ====================================================================
        # Sub-score 1: Velocity Score (mentions per minute)
        # ====================================================================
        # Max score at 10 mentions/min
        velocity_score = min(velocity_15m / VELOCITY_MAX_MENTIONS_PER_MIN, 1.0)

        # ====================================================================
        # Sub-score 2: Acceleration Score (velocity spike)
        # ====================================================================
        # Max score at 5x acceleration
        acceleration_score = min(abs(acceleration_15m) / ACCELERATION_MAX, 1.0)

        # ====================================================================
        # Sub-score 3: Author Diversity Score (unique authors)
        # ====================================================================
        # Max score at 50 unique authors
        author_diversity_score = min(unique_authors / AUTHOR_DIVERSITY_MAX, 1.0)

        # ====================================================================
        # Sub-score 4: Anomaly Score (z-score spike detection)
        # ====================================================================
        anomaly_score = 0.0
        anomaly_count_result = self.anomaly_detector.detect_anomaly(
            ticker, float(mention_count), "mention_count"
        )
        if anomaly_count_result["is_anomaly"]:
            anomaly_score = 1.0
        else:
            # Also check velocity anomaly
            anomaly_velocity_result = self.anomaly_detector.detect_anomaly(
                ticker, velocity_15m, "mention_velocity"
            )
            if anomaly_velocity_result["is_anomaly"]:
                anomaly_score = 1.0

        # ====================================================================
        # Sub-score 5: Cross-source Confirmation Score
        # ====================================================================
        confirmation_flag, confirmation_strength = self.confirmer.compute_confirmation_flag(ticker)

        # ====================================================================
        # Weighted Composite Score
        # ====================================================================
        gossip_score = (
            (velocity_score * GOSSIP_SCORE_WEIGHTS["velocity"]) +
            (acceleration_score * GOSSIP_SCORE_WEIGHTS["acceleration"]) +
            (author_diversity_score * GOSSIP_SCORE_WEIGHTS["author_diversity"]) +
            (anomaly_score * GOSSIP_SCORE_WEIGHTS["anomaly"]) +
            (confirmation_strength * GOSSIP_SCORE_WEIGHTS["confirmation"])
        )

        # Clamp to [0, 1]
        gossip_score = max(0.0, min(gossip_score, 1.0))

        # ====================================================================
        # Extract top phrases from recent events
        # ====================================================================
        top_phrases = []
        try:
            mentions_df = get_recent_mentions(ticker, minutes=60)
            phrase_counts = {}

            for idx, row in mentions_df.iterrows():
                text = row.get("text") or ""
                if text:
                    phrases = self.phrase_engine.extract_ngrams(text, n_range=(2, 3))
                    for phrase in phrases:
                        phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

            # Get top 10 phrases by frequency
            top_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            top_phrases = [phrase for phrase, count in top_phrases]

        except Exception as e:
            logger.debug(f"Error extracting top phrases for {ticker}: {e}")

        # ====================================================================
        # Build result dict
        # ====================================================================
        result = {
            "ticker": ticker,
            "computed_at": datetime.utcnow(),
            "gossip_score": gossip_score,
            "mention_count": mention_count,
            "mention_velocity": velocity_15m,
            "acceleration": acceleration_15m,
            "unique_authors": unique_authors,
            "top_phrases": str(top_phrases),  # Store as string for JSON serialization
            "velocity_metrics": str(metrics["windows"]),  # Store all window metrics
            "confirmation_flag": confirmation_flag,
            "alert_triggered": bool(anomaly_count_result["is_anomaly"]),
        }

        return result

    def score_all_active(self) -> list[dict]:
        """
        Score all active tickers and persist results.

        Computes scores, stores in database and Redis, returns sorted list.

        Returns:
            List of score dicts sorted by gossip_score descending
        """
        # Get all active tickers
        tickers = self.tracker.get_all_active_tickers(lookback_minutes=60)
        logger.info(f"Scoring {len(tickers)} active tickers")

        scores = []

        for ticker in tickers:
            try:
                score = self.compute_score(ticker)
                scores.append(score)

                # Store in database
                insert_gossip_score(score)

                # Store top scorer in Redis
                if score["gossip_score"] > 0.5:
                    self.redis_client.store_ticker_snapshot(
                        ticker,
                        {
                            "gossip_score": score["gossip_score"],
                            "mention_velocity": score["mention_velocity"],
                            "acceleration": score["acceleration"],
                            "unique_authors": score["unique_authors"],
                            "top_phrases": score["top_phrases"],
                            "confirmation_flag": score["confirmation_flag"],
                            "computed_at": score["computed_at"].isoformat(),
                        }
                    )

                    # Store alert if triggered
                    if score["alert_triggered"]:
                        self.redis_client.store_alert(
                            ticker,
                            {
                                "gossip_score": score["gossip_score"],
                                "mention_velocity": score["mention_velocity"],
                                "reason": "anomaly_detected",
                            }
                        )

            except Exception as e:
                logger.error(f"Error scoring {ticker}: {e}")

        # Sort by gossip_score descending
        scores.sort(key=lambda x: x["gossip_score"], reverse=True)

        logger.info(f"Scored {len(scores)} tickers. Top: {scores[0]['ticker'] if scores else 'N/A'}")
        return scores
