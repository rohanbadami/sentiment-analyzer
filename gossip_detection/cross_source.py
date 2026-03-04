"""
cross_source.py - Cross-source confirmation logic (Reddit x StockTwits).

Detects when the same phrases appear across multiple sources, increasing confidence
in the gossip signal.

Note: StockTwits ingestion is handled separately. This module works with the
schema when data is available.
"""

import logging
from datetime import datetime, timedelta
from sqlalchemy import text

from db_mysql import get_engine
from gossip_detection.phrase_engine import PhraseEngine

logger = logging.getLogger(__name__)


class CrossSourceConfirmer:
    """
    Detects phrase overlap and narrative consistency across sources.

    Currently focuses on Reddit. StockTwits support is placeholder for future integration.
    """

    def __init__(self, time_window_minutes: int = 30):
        """
        Initialize cross-source confirmer.

        Args:
            time_window_minutes: Window for proximity-based matching
        """
        self.time_window_minutes = time_window_minutes
        self.engine = get_engine()
        self.phrase_engine = PhraseEngine()

        if not self.engine:
            logger.error("Failed to initialize database engine")

    def _check_stocktwits_table_exists(self) -> bool:
        """
        Check if StockTwits events table exists in database.

        Returns:
            True if table exists, False otherwise
        """
        try:
            query = "SELECT 1 FROM stocktwits_events LIMIT 1"
            with self.engine.connect() as conn:
                conn.execute(text(query))
            return True
        except Exception:
            return False

    def find_phrase_overlap(
        self, ticker: str, window_minutes: int = 60
    ) -> list[dict]:
        """
        Find phrases appearing on both Reddit and StockTwits.

        Args:
            ticker: Stock ticker symbol
            window_minutes: Historical window to search

        Returns:
            List of overlap dicts with confirmation strength
        """
        overlaps = []

        # Get Reddit events for ticker
        try:
            query = """
            SELECT re.text, re.created_utc, re.upvotes
            FROM reddit_events re
            JOIN ticker_mentions tm ON re.id = tm.event_id
            WHERE tm.ticker = :ticker
            AND re.created_utc >= DATE_SUB(NOW(), INTERVAL :window MINUTE)
            """

            with self.engine.connect() as conn:
                result = conn.execute(
                    text(query),
                    {"ticker": ticker, "window": window_minutes}
                )
                reddit_rows = result.fetchall()

        except Exception as e:
            logger.error(f"Error querying Reddit events for {ticker}: {e}")
            reddit_rows = []

        if not reddit_rows:
            logger.debug(f"No Reddit events found for {ticker}")
            return []

        # Check if StockTwits table exists
        has_stocktwits = self._check_stocktwits_table_exists()

        # Get StockTwits events if available
        stocktwits_rows = []
        if has_stocktwits:
            try:
                query = """
                SELECT st.text, st.created_utc, st.upvotes
                FROM stocktwits_events st
                WHERE st.ticker = :ticker
                AND st.created_utc >= DATE_SUB(NOW(), INTERVAL :window MINUTE)
                """

                with self.engine.connect() as conn:
                    result = conn.execute(
                        text(query),
                        {"ticker": ticker, "window": window_minutes}
                    )
                    stocktwits_rows = result.fetchall()

            except Exception as e:
                logger.debug(f"Error querying StockTwits events: {e}")
                stocktwits_rows = []

        # If no StockTwits data, no cross-source confirmation possible
        if not stocktwits_rows:
            logger.debug(f"No StockTwits data available for {ticker}")
            return []

        # Extract phrases from each source
        reddit_phrases = {}  # phrase -> count, timestamps
        for row in reddit_rows:
            text, created_utc, upvotes = row
            if not text:
                continue

            phrases = self.phrase_engine.extract_ngrams(text, n_range=(2, 3))

            for phrase in phrases:
                if phrase not in reddit_phrases:
                    reddit_phrases[phrase] = {"count": 0, "first_seen": created_utc, "total_upvotes": 0}
                reddit_phrases[phrase]["count"] += 1
                reddit_phrases[phrase]["total_upvotes"] += upvotes

        stocktwits_phrases = {}
        for row in stocktwits_rows:
            text, created_utc, upvotes = row
            if not text:
                continue

            phrases = self.phrase_engine.extract_ngrams(text, n_range=(2, 3))

            for phrase in phrases:
                if phrase not in stocktwits_phrases:
                    stocktwits_phrases[phrase] = {"count": 0, "first_seen": created_utc, "total_upvotes": 0}
                stocktwits_phrases[phrase]["count"] += 1
                stocktwits_phrases[phrase]["total_upvotes"] += upvotes

        # Find overlaps
        common_phrases = set(reddit_phrases.keys()) & set(stocktwits_phrases.keys())

        for phrase in common_phrases:
            reddit_data = reddit_phrases[phrase]
            stocktwits_data = stocktwits_phrases[phrase]

            # Calculate confirmation strength based on counts and time proximity
            reddit_count = reddit_data["count"]
            stocktwits_count = stocktwits_data["count"]

            # Strength: normalized count product
            max_count = max(reddit_count, stocktwits_count)
            min_count = min(reddit_count, stocktwits_count)
            count_strength = (min_count / max_count) if max_count > 0 else 0

            # Time gap between first mentions
            time_gap = abs(
                (reddit_data["first_seen"] - stocktwits_data["first_seen"]).total_seconds()
            )

            # Penalize large time gaps (max penalty at 1 hour)
            time_strength = max(0, 1.0 - (time_gap / 3600.0))

            # Combined strength
            confirmation_strength = (count_strength * 0.6) + (time_strength * 0.4)

            overlaps.append({
                "phrase": phrase,
                "reddit_count": reddit_count,
                "stocktwits_count": stocktwits_count,
                "first_seen_reddit": reddit_data["first_seen"],
                "first_seen_stocktwits": stocktwits_data["first_seen"],
                "time_gap_seconds": int(time_gap),
                "confirmation_strength": confirmation_strength,
            })

        logger.info(f"{ticker}: found {len(overlaps)} overlapping phrases across sources")
        return overlaps

    def compute_confirmation_flag(self, ticker: str) -> tuple[bool, float]:
        """
        Compute cross-source confirmation for a ticker.

        Returns (flag: bool, strength: float).
        Flag is True if strong phrase overlap detected.
        Strength is max confirmation_strength across overlaps.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Tuple of (confirmation_flag, max_confirmation_strength)
        """
        overlaps = self.find_phrase_overlap(ticker)

        if not overlaps:
            return False, 0.0

        # Find max confirmation strength
        max_strength = max(o["confirmation_strength"] for o in overlaps)

        # Flag if any overlap has strength > 0.5
        flag = any(o["confirmation_strength"] > 0.5 for o in overlaps)

        return flag, max_strength
