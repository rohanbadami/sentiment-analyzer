"""
event_schema.py - Standardized event dataclass for Reddit posts and comments.

Provides RedditEvent dataclass with conversion methods from Pullpush.io and
Arctic Shift API responses.
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, Any


@dataclass
class RedditEvent:
    """
    Standardized representation of a Reddit post or comment.

    Fields are designed to be flattened for direct MySQL insertion via to_dict().
    """

    id: str  # Auto-generated primary key (not Reddit's ID)
    reddit_id: str  # Reddit's native ID (e.g., "t3_xyz" or "t1_xyz")
    source: str = "reddit"
    subreddit: str = ""
    author: str = ""
    author_karma: Optional[int] = None
    title: Optional[str] = None  # None for comments
    text: str = ""
    post_type: str = "post"  # "post" or "comment"
    created_utc: Optional[datetime] = None
    upvotes: int = 0
    num_comments: int = 0  # 0 for comments
    permalink: str = ""
    tickers: list[str] = field(default_factory=list)  # Extracted later, default empty
    engagement_score: float = 0.0  # Computed later
    raw_metadata: dict = field(default_factory=dict)  # Full Reddit JSON

    def to_dict(self) -> dict:
        """
        Convert to flat dictionary for MySQL insertion.

        Returns dict with snake_case keys ready for INSERT.
        """
        result = asdict(self)
        # Convert datetime to string for MySQL
        if self.created_utc:
            result['created_utc'] = self.created_utc.isoformat()
        # Convert lists to JSON strings for storage
        result['tickers'] = str(self.tickers)
        # Convert dict to JSON string
        result['raw_metadata'] = str(self.raw_metadata)
        return result

    @classmethod
    def from_pullpush_json(cls, data: dict) -> Optional['RedditEvent']:
        """
        Convert Pullpush.io API response to RedditEvent.

        Handles both posts (from /search/submission) and comments (from /search/comment).
        Returns None if content is deleted.

        Args:
            data: JSON dict from Pullpush API

        Returns:
            RedditEvent instance or None if deleted/removed
        """
        # Skip deleted/removed content
        author = data.get("author", "")
        if author == "[deleted]":
            return None

        # Determine post type from field presence
        is_post = "selftext" in data and "title" in data
        post_type = "post" if is_post else "comment"

        # Get text content
        if is_post:
            text_content = data.get("selftext", "")
            title = data.get("title", "")
        else:
            text_content = data.get("body", "")
            title = None

        # Skip if text is deleted/removed
        if text_content in ["[deleted]", "[removed]", ""]:
            return None

        # Build reddit_id from kind + id
        reddit_kind = "t3" if is_post else "t1"
        reddit_id = f"{reddit_kind}_{data.get('id', '')}"

        # Create event
        event = cls(
            id="",  # Will be auto-generated on DB insert
            reddit_id=reddit_id,
            source="reddit",
            subreddit=data.get("subreddit", ""),
            author=author,
            author_karma=data.get("author_cakeday_count") or data.get("score"),  # Fallback to post score
            title=title,
            text=text_content,
            post_type=post_type,
            created_utc=datetime.utcfromtimestamp(data.get("created_utc", 0)),
            upvotes=data.get("score", 0),
            num_comments=data.get("num_comments", 0) if is_post else 0,
            permalink=data.get("permalink", ""),
            raw_metadata=data,
        )

        return event

    @classmethod
    def from_arctic_shift_json(cls, data: dict) -> Optional['RedditEvent']:
        """
        Convert Arctic Shift API response to RedditEvent.

        Arctic Shift is an alternative archive; response format varies slightly from Pullpush.

        Args:
            data: JSON dict from Arctic Shift API

        Returns:
            RedditEvent instance or None if deleted/removed
        """
        # Skip deleted/removed content
        author = data.get("author", "")
        if author == "[deleted]":
            return None

        # Determine post type
        is_post = "selftext" in data and "title" in data
        post_type = "post" if is_post else "comment"

        # Get text content
        if is_post:
            text_content = data.get("selftext", "")
            title = data.get("title", "")
        else:
            text_content = data.get("body", "")
            title = None

        # Skip if text is deleted/removed
        if text_content in ["[deleted]", "[removed]", ""]:
            return None

        # Build reddit_id
        reddit_kind = "t3" if is_post else "t1"
        reddit_id = f"{reddit_kind}_{data.get('id', '')}"

        # Create event
        event = cls(
            id="",
            reddit_id=reddit_id,
            source="reddit",
            subreddit=data.get("subreddit", ""),
            author=author,
            author_karma=data.get("author_cakeday_count"),
            title=title,
            text=text_content,
            post_type=post_type,
            created_utc=datetime.utcfromtimestamp(data.get("created_utc", 0)),
            upvotes=data.get("score", 0),
            num_comments=data.get("num_comments", 0) if is_post else 0,
            permalink=data.get("permalink", ""),
            raw_metadata=data,
        )

        return event
