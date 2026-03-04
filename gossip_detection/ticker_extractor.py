"""
ticker_extractor.py - Extracts stock ticker mentions from Reddit text.

Uses pattern matching (cashtag $TSLA, bare uppercase) and validates against
finviz.csv US equity whitelist.
"""

import re
import csv
from pathlib import Path
from typing import Optional

import logging

logger = logging.getLogger(__name__)


class TickerExtractor:
    """
    Extracts and validates stock ticker mentions from Reddit posts/comments.

    Uses two patterns:
    1. Cashtag: $TSLA
    2. Bare uppercase: NVDA (if in whitelist)
    """

    def __init__(self, finviz_path: Optional[str] = None):
        """
        Initialize extractor with valid US equity tickers.

        Args:
            finviz_path: Path to finviz.csv. Defaults to root of repo.
        """
        self.valid_tickers = self._load_valid_tickers(finviz_path)
        self.common_words = self._build_exclusion_set()

    def _load_valid_tickers(self, finviz_path: Optional[str]) -> set[str]:
        """
        Load valid US equity tickers from finviz.csv.

        Args:
            finviz_path: Path to CSV file with "Ticker" column

        Returns:
            Set of uppercase ticker symbols
        """
        if finviz_path is None:
            finviz_path = Path(__file__).parent.parent / "finviz.csv"

        valid_tickers = set()

        try:
            with open(finviz_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ticker = row.get("Ticker", "").strip().upper()
                    if ticker:
                        valid_tickers.add(ticker)

            logger.info(f"Loaded {len(valid_tickers)} valid tickers from {finviz_path}")
        except FileNotFoundError:
            logger.error(f"finviz.csv not found at {finviz_path}")
        except Exception as e:
            logger.error(f"Error loading finviz.csv: {e}")

        return valid_tickers

    def _build_exclusion_set(self) -> set[str]:
        """
        Build set of common English words to exclude from bare ticker matching.

        Returns:
            Set of uppercase words
        """
        common_words = {
            "A", "I", "IT", "ARE", "FOR", "ALL", "CEO", "USA", "AI", "AM", "AN",
            "AT", "BE", "BY", "DO", "GO", "IF", "IN", "IS", "ME", "MY", "NO",
            "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE", "BUY", "SELL", "PUT",
            "CALL", "DD", "YOLO", "EDIT", "TL;DR", "OP", "ETA", "WSB", "HOD",
            "HODL", "PD", "RH", "FD", "IV", "ER", "IPO", "EPS", "PE", "ROE", "ROI",
            "ETF", "ETH", "BTC", "ATH", "GAP", "MA", "PM", "QQQ", "SPY", "IWM",
            "SQ", "FB", "TSM", "GS", "BAC", "MS", "JPM", "PD", "RH"
        }
        return common_words

    def extract_tickers(self, text: str) -> list[str]:
        """
        Extract stock ticker mentions from text.

        Patterns:
        1. Cashtag: $TSLA
        2. Bare uppercase: NVDA (if in whitelist and not a common word)

        Args:
            text: Reddit post/comment text

        Returns:
            Deduplicated list of valid ticker symbols
        """
        tickers = set()

        # Pattern 1: Cashtag ($TSLA)
        cashtag_pattern = r'\$([A-Z]{1,5})\b'
        for match in re.finditer(cashtag_pattern, text):
            ticker = match.group(1).upper()
            if ticker in self.valid_tickers:
                tickers.add(ticker)

        # Pattern 2: Bare uppercase (NVDA)
        uppercase_pattern = r'\b([A-Z]{2,5})\b'
        for match in re.finditer(uppercase_pattern, text):
            ticker = match.group(1).upper()

            # Skip if common word or not in whitelist
            if ticker in self.common_words or ticker not in self.valid_tickers:
                continue

            tickers.add(ticker)

        return sorted(list(tickers))

    def bucket_events(
        self, events: list, bucket_minutes: int = 5
    ) -> dict[tuple[str, int], list]:
        """
        Group events by (ticker, time_bucket).

        Args:
            events: List of RedditEvent objects
            bucket_minutes: Time bucket size in minutes

        Returns:
            Dict: {(ticker, bucket_timestamp_utc): [list of events]}
        """
        buckets = {}

        for event in events:
            # Skip events without tickers
            if not event.tickers:
                continue

            # Compute bucket timestamp
            event_utc = int(event.created_utc.timestamp())
            bucket_size_seconds = bucket_minutes * 60
            bucket_timestamp = (event_utc // bucket_size_seconds) * bucket_size_seconds

            # Add to all relevant ticker buckets
            for ticker in event.tickers:
                key = (ticker, bucket_timestamp)
                if key not in buckets:
                    buckets[key] = []
                buckets[key].append(event)

        return buckets
