"""
redis_client.py - Redis client for hot storage of gossip detection data.

Stores latest Reddit pull, ticker snapshots, and alerts with configurable TTL.
"""

import json
import logging
from datetime import datetime
from typing import Optional

import redis

from gossip_detection.config import (
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    REDIS_PREFIX_LATEST_PULL,
    REDIS_PREFIX_LATEST_PULL_TS,
    REDIS_PREFIX_TICKER_SNAPSHOT,
    REDIS_PREFIX_ALERTS,
    REDIS_TTL_LATEST_PULL,
    REDIS_TTL_TICKER_SNAPSHOT,
)

logger = logging.getLogger(__name__)


class GossipRedisClient:
    """
    Redis client for hot storage of gossip detection metrics and alerts.

    Implements short-lived caches for real-time dashboard updates and alerts.
    """

    def __init__(self):
        """
        Initialize Redis client with configured host/port/db.

        Raises connection on failure but doesn't crash initialization.
        """
        try:
            self.client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True,
            )
            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.client = None

    def _is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self.client is not None

    def store_latest_pull(self, events: list[dict], pull_timestamp: datetime) -> bool:
        """
        Store latest Reddit pull in Redis with TTL.

        Args:
            events: List of event dicts (from RedditEvent.to_dict())
            pull_timestamp: When the pull occurred

        Returns:
            True if stored successfully, False otherwise
        """
        if not self._is_connected():
            return False

        try:
            # Serialize events to JSON
            events_json = json.dumps(events)

            # Store events
            self.client.setex(
                REDIS_PREFIX_LATEST_PULL,
                REDIS_TTL_LATEST_PULL,
                events_json,
            )

            # Store timestamp
            ts_iso = pull_timestamp.isoformat()
            self.client.setex(
                REDIS_PREFIX_LATEST_PULL_TS,
                REDIS_TTL_LATEST_PULL,
                ts_iso,
            )

            logger.debug(f"Stored latest pull: {len(events)} events")
            return True

        except Exception as e:
            logger.error(f"Failed to store latest pull: {e}")
            return False

    def get_latest_pull(self) -> tuple[list[dict], Optional[datetime]]:
        """
        Retrieve latest Reddit pull from Redis.

        Returns:
            Tuple of (events_list, timestamp) or ([], None) if expired/missing
        """
        if not self._is_connected():
            return [], None

        try:
            events_json = self.client.get(REDIS_PREFIX_LATEST_PULL)
            ts_iso = self.client.get(REDIS_PREFIX_LATEST_PULL_TS)

            if not events_json or not ts_iso:
                return [], None

            events = json.loads(events_json)
            timestamp = datetime.fromisoformat(ts_iso)

            return events, timestamp

        except Exception as e:
            logger.error(f"Failed to get latest pull: {e}")
            return [], None

    def store_ticker_snapshot(self, ticker: str, metrics: dict) -> bool:
        """
        Store current metrics for a ticker (hot snapshot).

        Args:
            ticker: Stock ticker symbol
            metrics: Dict of metrics (gossip_score, velocity, etc.)

        Returns:
            True if stored successfully
        """
        if not self._is_connected():
            return False

        try:
            key = f"{REDIS_PREFIX_TICKER_SNAPSHOT}:{ticker}"
            metrics_json = json.dumps(metrics)

            self.client.setex(
                key,
                REDIS_TTL_TICKER_SNAPSHOT,
                metrics_json,
            )

            logger.debug(f"Stored snapshot for {ticker}")
            return True

        except Exception as e:
            logger.error(f"Failed to store ticker snapshot: {e}")
            return False

    def get_ticker_snapshot(self, ticker: str) -> Optional[dict]:
        """
        Retrieve current metrics for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict of metrics or None if expired/missing
        """
        if not self._is_connected():
            return None

        try:
            key = f"{REDIS_PREFIX_TICKER_SNAPSHOT}:{ticker}"
            metrics_json = self.client.get(key)

            if not metrics_json:
                return None

            return json.loads(metrics_json)

        except Exception as e:
            logger.error(f"Failed to get ticker snapshot: {e}")
            return None

    def store_alert(self, ticker: str, alert_data: dict) -> bool:
        """
        Store an alert to the Redis alerts list (LIFO queue).

        Args:
            ticker: Stock ticker symbol
            alert_data: Dict with alert details (gossip_score, reason, etc.)

        Returns:
            True if stored successfully
        """
        if not self._is_connected():
            return False

        try:
            alert_with_ts = alert_data.copy()
            alert_with_ts['timestamp'] = datetime.utcnow().isoformat()
            alert_with_ts['ticker'] = ticker

            alert_json = json.dumps(alert_with_ts)

            # LPUSH adds to head; keep only last 100
            self.client.lpush(REDIS_PREFIX_ALERTS, alert_json)
            self.client.ltrim(REDIS_PREFIX_ALERTS, 0, 99)

            logger.debug(f"Stored alert for {ticker}")
            return True

        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
            return False

    def get_recent_alerts(self, count: int = 20) -> list[dict]:
        """
        Retrieve recent alerts from the queue.

        Args:
            count: Number of alerts to retrieve (max 100)

        Returns:
            List of alert dicts (most recent first)
        """
        if not self._is_connected():
            return []

        try:
            count = min(count, 100)  # Cap at 100
            alerts_json_list = self.client.lrange(REDIS_PREFIX_ALERTS, 0, count - 1)

            alerts = []
            for alert_json in alerts_json_list:
                try:
                    alert = json.loads(alert_json)
                    alerts.append(alert)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse alert JSON: {alert_json}")

            return alerts

        except Exception as e:
            logger.error(f"Failed to get recent alerts: {e}")
            return []
