"""Tests for event_schema.py - RedditEvent mapping and validation."""

import pytest
from datetime import datetime
from gossip_detection.event_schema import RedditEvent


class TestRedditEventSchema:
    """Test RedditEvent dataclass."""

    def test_event_to_dict(self):
        """Test RedditEvent.to_dict() produces flat dict."""
        event = RedditEvent(
            id="1",
            reddit_id="t3_abc123",
            subreddit="wallstreetbets",
            author="testuser",
            text="$TSLA is pumping!",
            post_type="post",
            created_utc=datetime(2024, 1, 1, 12, 0, 0),
            tickers=["TSLA"],
        )

        result = event.to_dict()

        assert result["reddit_id"] == "t3_abc123"
        assert result["tickers"] == "['TSLA']"  # Converted to string
        assert "created_utc" in result
        assert isinstance(result["created_utc"], str)

    def test_from_pullpush_json_post(self):
        """Test parsing Pullpush post response."""
        pullpush_data = {
            "id": "abc123",
            "title": "TSLA Moon Shot",
            "selftext": "Bullish on Tesla",
            "author": "testuser",
            "subreddit": "wallstreetbets",
            "score": 100,
            "num_comments": 50,
            "created_utc": 1704110400,  # 2024-01-01 12:00:00 UTC
            "permalink": "/r/wallstreetbets/comments/abc123/",
        }

        event = RedditEvent.from_pullpush_json(pullpush_data)

        assert event is not None
        assert event.reddit_id == "t3_abc123"
        assert event.post_type == "post"
        assert event.title == "TSLA Moon Shot"
        assert event.text == "Bullish on Tesla"
        assert event.author == "testuser"
        assert event.upvotes == 100

    def test_from_pullpush_json_comment(self):
        """Test parsing Pullpush comment response."""
        pullpush_data = {
            "id": "def456",
            "body": "Great analysis!",
            "author": "commenter",
            "subreddit": "stocks",
            "score": 25,
            "created_utc": 1704110400,
            "permalink": "/r/stocks/comments/xyz/",
        }

        event = RedditEvent.from_pullpush_json(pullpush_data)

        assert event is not None
        assert event.post_type == "comment"
        assert event.title is None
        assert event.text == "Great analysis!"
        assert event.num_comments == 0

    def test_from_pullpush_deleted_content_author(self):
        """Test that deleted content (author=[deleted]) is skipped."""
        pullpush_data = {
            "id": "abc123",
            "title": "Test",
            "selftext": "Content",
            "author": "[deleted]",
            "subreddit": "wallstreetbets",
            "score": 100,
            "created_utc": 1704110400,
            "permalink": "/r/wallstreetbets/comments/abc123/",
        }

        event = RedditEvent.from_pullpush_json(pullpush_data)
        assert event is None

    def test_from_pullpush_deleted_content_text(self):
        """Test that deleted content (selftext=[deleted]) is skipped."""
        pullpush_data = {
            "id": "abc123",
            "title": "Test",
            "selftext": "[deleted]",
            "author": "testuser",
            "subreddit": "wallstreetbets",
            "score": 100,
            "created_utc": 1704110400,
            "permalink": "/r/wallstreetbets/comments/abc123/",
        }

        event = RedditEvent.from_pullpush_json(pullpush_data)
        assert event is None

    def test_from_pullpush_removed_content(self):
        """Test that removed content (selftext=[removed]) is skipped."""
        pullpush_data = {
            "id": "abc123",
            "title": "Test",
            "selftext": "[removed]",
            "author": "testuser",
            "subreddit": "wallstreetbets",
            "score": 100,
            "created_utc": 1704110400,
            "permalink": "/r/wallstreetbets/comments/abc123/",
        }

        event = RedditEvent.from_pullpush_json(pullpush_data)
        assert event is None

    def test_from_arctic_shift_json(self):
        """Test parsing Arctic Shift response (similar schema to Pullpush)."""
        arctic_data = {
            "id": "xyz789",
            "title": "AMD News",
            "selftext": "Positive earnings",
            "author": "trader",
            "subreddit": "stocks",
            "score": 50,
            "created_utc": 1704110400,
            "permalink": "/r/stocks/comments/xyz789/",
        }

        event = RedditEvent.from_arctic_shift_json(arctic_data)

        assert event is not None
        assert event.reddit_id == "t3_xyz789"
        assert event.text == "Positive earnings"
        assert event.author == "trader"
