"""
anomaly_detector.py - Detects anomalous spikes in mention activity.

Uses z-score based anomaly detection with configurable thresholds and baseline
windows for historical context.
"""

import logging
import math
from datetime import datetime, timedelta
from sqlalchemy import text

from db_mysql import get_engine
from gossip_detection.config import Z_SCORE_THRESHOLD, MIN_BASELINE_POINTS
from gossip_detection.rolling_tracker import RollingTracker

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Detects anomalous spikes in mention activity using z-score and thresholds.

    Compares current metrics against historical baseline to identify unusual patterns.
    """

    def __init__(self, z_threshold: float = Z_SCORE_THRESHOLD,
                 min_baseline_points: int = MIN_BASELINE_POINTS):
        """
        Initialize anomaly detector.

        Args:
            z_threshold: Z-score threshold for anomaly (default 2.5)
            min_baseline_points: Minimum baseline data points required
        """
        self.z_threshold = z_threshold
        self.min_baseline_points = min_baseline_points
        self.engine = get_engine()

        if not self.engine:
            logger.error("Failed to initialize database engine")

    def compute_baseline(
        self, ticker: str, metric_name: str, lookback_hours: int = 24
    ) -> tuple[float, float]:
        """
        Compute baseline (mean, std_dev) for a metric over historical window.

        Args:
            ticker: Stock ticker symbol
            metric_name: Metric name (gossip_score, mention_velocity, etc.)
            lookback_hours: Historical lookback window in hours

        Returns:
            Tuple of (mean, std_dev) or (None, None) if insufficient data
        """
        try:
            # Query historical gossip_scores for this metric
            query = """
            SELECT {metric}
            FROM gossip_scores
            WHERE ticker = :ticker
            AND computed_at >= DATE_SUB(NOW(), INTERVAL :hours HOUR)
            ORDER BY computed_at
            """

            # Map metric names to column names
            column_map = {
                "gossip_score": "gossip_score",
                "mention_velocity": "mention_velocity",
                "mention_count": "mention_count",
                "acceleration": "acceleration",
                "unique_authors": "unique_authors",
            }

            column = column_map.get(metric_name, metric_name)
            query = query.format(metric=column)

            with self.engine.connect() as conn:
                result = conn.execute(
                    text(query),
                    {"ticker": ticker, "hours": lookback_hours}
                )
                rows = result.fetchall()

            if len(rows) < self.min_baseline_points:
                logger.debug(
                    f"Insufficient baseline for {ticker}:{metric_name} "
                    f"({len(rows)} < {self.min_baseline_points})"
                )
                return None, None

            values = [float(row[0]) for row in rows if row[0] is not None]

            if not values:
                return None, None

            # Compute mean
            mean = sum(values) / len(values)

            # Compute standard deviation
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            std_dev = math.sqrt(variance)

            logger.debug(f"Baseline for {ticker}:{metric_name} -> mean={mean:.3f}, std={std_dev:.3f}")
            return mean, std_dev

        except Exception as e:
            logger.error(f"Error computing baseline for {ticker}:{metric_name}: {e}")
            return None, None

    def detect_anomaly(
        self, ticker: str, current_value: float, metric_name: str
    ) -> dict:
        """
        Detect if current value is anomalous compared to baseline.

        Args:
            ticker: Stock ticker symbol
            current_value: Current metric value
            metric_name: Name of the metric

        Returns:
            Dict with anomaly detection results
        """
        mean, std = self.compute_baseline(ticker, metric_name)

        result = {
            "ticker": ticker,
            "metric": metric_name,
            "current_value": current_value,
            "baseline_mean": mean,
            "baseline_std": std,
            "z_score": None,
            "is_anomaly": False,
            "direction": None,
            "detected_at": datetime.utcnow(),
        }

        # If insufficient baseline, cannot detect anomaly
        if mean is None or std is None:
            result["is_anomaly"] = False
            return result

        # Avoid division by zero
        if std == 0:
            logger.debug(f"Zero std_dev for {ticker}:{metric_name}, skipping anomaly check")
            result["is_anomaly"] = False
            return result

        # Compute z-score
        z_score = (current_value - mean) / std
        result["z_score"] = z_score

        # Check if anomalous
        if abs(z_score) >= self.z_threshold:
            result["is_anomaly"] = True
            result["direction"] = "spike" if z_score > 0 else "drop"

        return result

    def scan_all_tickers(self, tracker: RollingTracker) -> list[dict]:
        """
        Scan all active tickers for anomalies.

        Checks mention_count and velocity at 15m window.

        Args:
            tracker: RollingTracker instance

        Returns:
            List of anomaly dicts where is_anomaly == True
        """
        anomalies = []

        # Get all active tickers
        tickers = tracker.get_all_active_tickers(lookback_minutes=60)

        for ticker in tickers:
            try:
                # Get 15m metrics
                count_15m = tracker.get_mention_counts(ticker, 15)
                velocity_15m = tracker.get_mention_velocity(ticker, 15)

                # Check mention count for anomaly
                anomaly_count = self.detect_anomaly(ticker, float(count_15m), "mention_count")
                if anomaly_count["is_anomaly"]:
                    anomalies.append(anomaly_count)

                # Check velocity for anomaly
                anomaly_velocity = self.detect_anomaly(ticker, velocity_15m, "mention_velocity")
                if anomaly_velocity["is_anomaly"]:
                    anomalies.append(anomaly_velocity)

            except Exception as e:
                logger.error(f"Error scanning {ticker} for anomalies: {e}")

        logger.info(f"Scanned {len(tickers)} tickers, found {len(anomalies)} anomalies")
        return anomalies
