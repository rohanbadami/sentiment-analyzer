"""Tests for ticker_extractor.py - Ticker extraction and validation."""

import pytest
from pathlib import Path
from gossip_detection.ticker_extractor import TickerExtractor


class TestTickerExtractor:
    """Test TickerExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create TickerExtractor instance."""
        return TickerExtractor()

    def test_cashtag_extraction(self, extractor):
        """Test extraction of $TICKER syntax."""
        text = "$TSLA is pumping! $NVDA also up."
        result = extractor.extract_tickers(text)

        assert "TSLA" in result
        assert "NVDA" in result
        assert len(result) == 2

    def test_bare_uppercase_extraction(self, extractor):
        """Test extraction of bare uppercase tickers."""
        text = "AAPL and MSFT are trading well today"
        result = extractor.extract_tickers(text)

        assert "AAPL" in result
        assert "MSFT" in result

    def test_common_word_exclusion(self, extractor):
        """Test that common words are excluded."""
        text = "I think AI is great for AMD"
        result = extractor.extract_tickers(text)

        # "I" and "AI" should be excluded as common words
        assert "I" not in result
        assert "AI" not in result
        assert "AMD" in result

    def test_multiple_tickers_single_post(self, extractor):
        """Test extraction of multiple tickers."""
        text = "$AAPL $MSFT $GOOGL $AMZN are the tech giants"
        result = extractor.extract_tickers(text)

        assert len(result) == 4
        assert set(result) == {"AAPL", "MSFT", "GOOGL", "AMZN"}

    def test_duplicate_removal(self, extractor):
        """Test that duplicate tickers are deduplicated."""
        text = "$TSLA $TSLA Tesla stock is great TSLA"
        result = extractor.extract_tickers(text)

        assert result.count("TSLA") == 1  # Only one occurrence

    def test_invalid_ticker_filtering(self, extractor):
        """Test that invalid tickers (not in whitelist) are filtered."""
        text = "$XYZNOTREAL is not a ticker"
        result = extractor.extract_tickers(text)

        assert "XYZNOTREAL" not in result

    def test_cashtag_uppercase_required(self, extractor):
        """Test that cashtag pattern requires uppercase letters."""
        text = "$TSLA is a valid cashtag"
        result = extractor.extract_tickers(text)

        # Should match uppercase cashtag
        assert "TSLA" in result

        # Lowercase cashtags are not matched by the regex pattern
        text_lower = "$tsla should not match the pattern"
        result_lower = extractor.extract_tickers(text_lower)
        assert "TSLA" not in result_lower

    def test_bucket_events_by_time(self, extractor):
        """Test bucketing of events by time window."""
        from datetime import datetime
        from gossip_detection.event_schema import RedditEvent

        # Create test events
        event1 = RedditEvent(
            id="1",
            reddit_id="t3_1",
            subreddit="test",
            author="user1",
            text="$TSLA",
            created_utc=datetime(2024, 1, 1, 12, 0, 0),
            tickers=["TSLA"],
        )

        event2 = RedditEvent(
            id="2",
            reddit_id="t3_2",
            subreddit="test",
            author="user2",
            text="$TSLA",
            created_utc=datetime(2024, 1, 1, 12, 3, 0),  # 3 minutes later
            tickers=["TSLA"],
        )

        events = [event1, event2]
        buckets = extractor.bucket_events(events, bucket_minutes=5)

        # Both should be in same 5-minute bucket
        assert len(buckets) == 1
        key = list(buckets.keys())[0]
        assert key[0] == "TSLA"  # Ticker
        assert len(buckets[key]) == 2  # Two events

    def test_bucket_events_different_buckets(self, extractor):
        """Test that events in different time windows go to different buckets."""
        from datetime import datetime
        from gossip_detection.event_schema import RedditEvent

        # Create test events 10 minutes apart
        event1 = RedditEvent(
            id="1",
            reddit_id="t3_1",
            subreddit="test",
            author="user1",
            text="$NVDA",
            created_utc=datetime(2024, 1, 1, 12, 0, 0),
            tickers=["NVDA"],
        )

        event2 = RedditEvent(
            id="2",
            reddit_id="t3_2",
            subreddit="test",
            author="user2",
            text="$NVDA",
            created_utc=datetime(2024, 1, 1, 12, 10, 0),  # 10 minutes later
            tickers=["NVDA"],
        )

        events = [event1, event2]
        buckets = extractor.bucket_events(events, bucket_minutes=5)

        # Should be in 2 different 5-minute buckets
        assert len(buckets) == 2
