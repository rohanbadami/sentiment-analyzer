"""
rolling_tracker.py - Tracks mention metrics over rolling time windows.

Computes mention counts, velocity, acceleration, and author diversity for
configurable rolling windows (5m, 15m, 60m).
"""

import logging
from datetime import datetime, timedelta
from sqlalchemy import text

from db_mysql import get_engine
from gossip_detection.config import ROLLING_WINDOWS

logger = logging.getLogger(__name__)


class RollingTracker:
    """
    Tracks mention activity metrics over rolling time windows.

    Windows: 5, 15, 60 minutes (configurable).
    Computes velocity (mentions/min) and acceleration (velocity_now / velocity_prev).
    """

    def __init__(self):
        """Initialize tracker with default windows."""
        self.windows = ROLLING_WINDOWS  # [5, 15, 60] minutes
        self.engine = get_engine()

        if not self.engine:
            logger.error("Failed to initialize database engine")

    def get_mention_counts(self, ticker: str, window_minutes: int) -> int:
        """
        Get mention count for ticker within rolling window.

        Args:
            ticker: Stock ticker symbol
            window_minutes: Window size in minutes

        Returns:
            Count of mentions in window
        """
        try:
            query = """
            SELECT COUNT(*) as count FROM ticker_mentions
            WHERE ticker = :ticker
            AND mentioned_at >= DATE_SUB(NOW(), INTERVAL :minutes MINUTE)
            """

            with self.engine.connect() as conn:
                result = conn.execute(
                    text(query),
                    {"ticker": ticker, "minutes": window_minutes}
                )
                row = result.fetchone()
                return row[0] if row else 0

        except Exception as e:
            logger.error(f"Error getting mention counts for {ticker}: {e}")
            return 0

    def get_mention_velocity(self, ticker: str, window_minutes: int) -> float:
        """
        Get mention velocity (mentions per minute).

        Args:
            ticker: Stock ticker symbol
            window_minutes: Window size in minutes

        Returns:
            Mentions per minute (float)
        """
        count = self.get_mention_counts(ticker, window_minutes)
        if window_minutes <= 0:
            return 0.0

        velocity = count / float(window_minutes)
        return velocity

    def get_acceleration(self, ticker: str, window_minutes: int) -> float:
        """
        Get acceleration: (velocity_current - velocity_previous) / velocity_previous.

        Compares velocity in current window vs previous window of same size.

        Args:
            ticker: Stock ticker symbol
            window_minutes: Window size in minutes

        Returns:
            Acceleration ratio (0.0 if velocity_previous == 0)
        """
        # Get velocity in current window
        velocity_current = self.get_mention_velocity(ticker, window_minutes)

        # Get velocity in previous window (one period back)
        # This requires summing mentions from [2*window, window] period
        try:
            query = """
            SELECT COUNT(*) as count FROM ticker_mentions
            WHERE ticker = :ticker
            AND mentioned_at >= DATE_SUB(NOW(), INTERVAL :window_past MINUTE)
            AND mentioned_at < DATE_SUB(NOW(), INTERVAL :window_present MINUTE)
            """

            with self.engine.connect() as conn:
                result = conn.execute(
                    text(query),
                    {
                        "ticker": ticker,
                        "window_past": 2 * window_minutes,
                        "window_present": window_minutes,
                    }
                )
                row = result.fetchone()
                count_previous = row[0] if row else 0

            velocity_previous = count_previous / float(window_minutes)

            # Compute acceleration
            if velocity_previous == 0:
                return 0.0

            acceleration = (velocity_current - velocity_previous) / velocity_previous
            return acceleration

        except Exception as e:
            logger.error(f"Error computing acceleration for {ticker}: {e}")
            return 0.0

    def get_unique_authors(self, ticker: str, window_minutes: int) -> int:
        """
        Get count of unique authors mentioning ticker in window.

        Args:
            ticker: Stock ticker symbol
            window_minutes: Window size in minutes

        Returns:
            Count of unique authors
        """
        try:
            query = """
            SELECT COUNT(DISTINCT re.author) as count FROM ticker_mentions tm
            JOIN reddit_events re ON tm.event_id = re.id
            WHERE tm.ticker = :ticker
            AND tm.mentioned_at >= DATE_SUB(NOW(), INTERVAL :minutes MINUTE)
            """

            with self.engine.connect() as conn:
                result = conn.execute(
                    text(query),
                    {"ticker": ticker, "minutes": window_minutes}
                )
                row = result.fetchone()
                return row[0] if row else 0

        except Exception as e:
            logger.error(f"Error getting unique authors for {ticker}: {e}")
            return 0

    def compute_all_metrics(self, ticker: str) -> dict:
        """
        Compute all metrics for ticker across all windows.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict with metrics for each window
        """
        result = {
            "ticker": ticker,
            "computed_at": datetime.utcnow(),
            "windows": {},
        }

        for window in self.windows:
            window_key = f"{window}m"

            count = self.get_mention_counts(ticker, window)
            velocity = self.get_mention_velocity(ticker, window)
            acceleration = self.get_acceleration(ticker, window)
            unique_authors = self.get_unique_authors(ticker, window)

            result["windows"][window_key] = {
                "count": count,
                "velocity": velocity,
                "acceleration": acceleration,
                "unique_authors": unique_authors,
            }

        return result

    def get_all_active_tickers(self, lookback_minutes: int = 60) -> list[str]:
        """
        Get list of all tickers with recent mentions.

        Args:
            lookback_minutes: Lookback window in minutes

        Returns:
            Sorted list of ticker symbols
        """
        try:
            query = """
            SELECT DISTINCT ticker FROM ticker_mentions
            WHERE mentioned_at >= DATE_SUB(NOW(), INTERVAL :minutes MINUTE)
            ORDER BY ticker
            """

            with self.engine.connect() as conn:
                result = conn.execute(
                    text(query),
                    {"minutes": lookback_minutes}
                )
                rows = result.fetchall()
                return [row[0] for row in rows if row[0]]

        except Exception as e:
            logger.error(f"Error getting active tickers: {e}")
            return []
