"""
reddit_ingestion.py - Reddit poller using Pullpush.io and Arctic Shift APIs.

Fetches recent posts and comments from configured subreddits. Free, no OAuth required.
Implements retry logic and graceful fallback between APIs.
"""

import requests
import logging
import json
import time
import random
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from gossip_detection.config import (
    PULLPUSH_BASE_URL,
    ARCTIC_SHIFT_BASE_URL,
    SUBREDDITS,
    MAX_WORKERS,
    CHECKPOINT_FILE,
)
from gossip_detection.event_schema import RedditEvent


logger = logging.getLogger(__name__)


class RedditPoller:
    """
    Fetches Reddit posts and comments from Pullpush.io / Arctic Shift.

    Implements exponential backoff, jitter, and checkpoint-based resumable polling.
    """

    def __init__(self, subreddits: list[str]):
        """
        Initialize poller with session and retry strategy.

        Args:
            subreddits: List of subreddit names to monitor
        """
        self.subreddits = subreddits

        # Create session with Retry strategy (following phase1_headline_scraper.py pattern)
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Load last poll timestamp from checkpoint
        self.last_poll_utc = self._load_checkpoint()

    def _load_checkpoint(self) -> int:
        """
        Load last poll timestamp from JSON checkpoint file.

        Returns:
            Unix timestamp (UTC). Defaults to 1 hour ago if not found.
        """
        checkpoint_path = Path(CHECKPOINT_FILE)
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, "r") as f:
                    data = json.load(f)
                    ts = data.get("last_poll_utc")
                    if ts:
                        logger.info(f"Loaded checkpoint: last_poll_utc={ts}")
                        return ts
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")

        # Default: 1 hour ago
        default_ts = int(time.time()) - 3600
        logger.info(f"No checkpoint found. Defaulting to 1 hour ago: {default_ts}")
        return default_ts

    def _save_checkpoint(self, utc_timestamp: int):
        """Save last poll timestamp to checkpoint file."""
        checkpoint_path = Path(CHECKPOINT_FILE)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(checkpoint_path, "w") as f:
                json.dump({"last_poll_utc": utc_timestamp}, f)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def fetch_recent_posts(self, subreddit: str, after_utc: int) -> list[RedditEvent]:
        """
        Fetch recent posts from Pullpush, fallback to Arctic Shift.

        Args:
            subreddit: Subreddit name (without /r/)
            after_utc: Unix timestamp—fetch posts created after this time

        Returns:
            List of RedditEvent objects (posts only)
        """
        events = []

        # Try Pullpush first
        try:
            url = f"{PULLPUSH_BASE_URL}/search/submission"
            params = {
                "subreddit": subreddit,
                "after": after_utc,
                "sort": "desc",
                "size": 100,
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            if subreddit == self.subreddits[0]:
                logger.info(f"Pullpush posts response keys: {list(data.keys())}, status={response.status_code}")
            raw_count = len(data.get("data", []))
            for item in data.get("data", []):
                event = RedditEvent.from_pullpush_json(item)
                if event:
                    events.append(event)

            logger.info(f"Pullpush posts [{subreddit}]: raw={raw_count}, parsed={len(events)}")
            return events

        except Exception as e:
            logger.warning(f"Pullpush failed for {subreddit}: {e}. Trying Arctic Shift...")

        # Fallback to Arctic Shift
        try:
            url = f"{ARCTIC_SHIFT_BASE_URL}/posts/search"
            params = {
                "subreddit": subreddit,
                "after": after_utc,
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            raw_count = len(data.get("data", []))
            for item in data.get("data", []):
                event = RedditEvent.from_arctic_shift_json(item)
                if event:
                    events.append(event)

            logger.info(f"Arctic Shift posts [{subreddit}]: raw={raw_count}, parsed={len(events)}")
            return events

        except Exception as e:
            logger.error(f"Arctic Shift also failed for {subreddit}: {e}")
            return []

    def fetch_recent_comments(self, subreddit: str, after_utc: int) -> list[RedditEvent]:
        """
        Fetch recent comments from Pullpush, fallback to Arctic Shift.

        Args:
            subreddit: Subreddit name (without /r/)
            after_utc: Unix timestamp—fetch comments created after this time

        Returns:
            List of RedditEvent objects (comments only)
        """
        events = []

        # Try Pullpush first
        try:
            url = f"{PULLPUSH_BASE_URL}/search/comment"
            params = {
                "subreddit": subreddit,
                "after": after_utc,
                "sort": "desc",
                "size": 100,
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            raw_count = len(data.get("data", []))
            for item in data.get("data", []):
                event = RedditEvent.from_pullpush_json(item)
                if event:
                    events.append(event)

            logger.info(f"Pullpush comments [{subreddit}]: raw={raw_count}, parsed={len(events)}")
            return events

        except Exception as e:
            logger.warning(f"Pullpush failed for {subreddit} (comments): {e}. Trying Arctic Shift...")

        # Fallback to Arctic Shift
        try:
            url = f"{ARCTIC_SHIFT_BASE_URL}/comments/search"
            params = {
                "subreddit": subreddit,
                "after": after_utc,
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            raw_count = len(data.get("data", []))
            for item in data.get("data", []):
                event = RedditEvent.from_arctic_shift_json(item)
                if event:
                    events.append(event)

            logger.info(f"Arctic Shift comments [{subreddit}]: raw={raw_count}, parsed={len(events)}")
            return events

        except Exception as e:
            logger.error(f"Arctic Shift also failed for {subreddit} (comments): {e}")
            return []

    def _poll_subreddit(self, subreddit: str, after_utc: int) -> list[RedditEvent]:
        """
        Poll a single subreddit for posts and comments.

        Internal helper for parallel execution.

        Args:
            subreddit: Subreddit name
            after_utc: Fetch data after this timestamp

        Returns:
            Combined list of posts and comments
        """
        all_events = []

        # Fetch posts
        posts = self.fetch_recent_posts(subreddit, after_utc)
        all_events.extend(posts)

        # Small delay between posts and comments fetch
        time.sleep(random.uniform(0.3, 0.8))

        # Fetch comments
        comments = self.fetch_recent_comments(subreddit, after_utc)
        all_events.extend(comments)

        # Random jitter to avoid hammering API
        time.sleep(random.uniform(0.3, 0.8))

        return all_events

    def poll_all(self, since_utc: int) -> list[RedditEvent]:
        """
        Poll all configured subreddits in parallel.

        Args:
            since_utc: Fetch events created after this Unix timestamp

        Returns:
            Flat list of all RedditEvents from all subreddits, sorted by created_utc
        """
        all_events = []
        poll_start = time.time()

        logger.info(f"Starting poll cycle for {len(self.subreddits)} subreddits...")

        # Use ThreadPoolExecutor for parallel subreddit polling
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(self._poll_subreddit, sub, since_utc): sub
                for sub in self.subreddits
            }

            for future in as_completed(futures):
                subreddit = futures[future]
                try:
                    events = future.result()
                    all_events.extend(events)
                    logger.debug(f"{subreddit}: fetched {len(events)} events")
                except Exception as e:
                    logger.error(f"Error polling {subreddit}: {e}")

        # Sort by created_utc
        all_events.sort(key=lambda e: e.created_utc or datetime.utcnow())

        poll_end = time.time()
        cycle_time_ms = (poll_end - poll_start) * 1000

        # Log cycle stats
        newest_event_ts = all_events[-1].created_utc if all_events else None
        lag_seconds = (time.time() - newest_event_ts.timestamp()) if newest_event_ts else None

        lag_str = f"{lag_seconds:.0f}s" if lag_seconds is not None else "N/A"
        logger.info(
            f"Poll cycle complete: "
            f"{len(all_events)} events fetched, "
            f"cycle_time={cycle_time_ms:.0f}ms, "
            f"lag={lag_str}"
        )

        # Update checkpoint
        if all_events:
            newest_utc = int(all_events[-1].created_utc.timestamp())
            self._save_checkpoint(newest_utc)
            self.last_poll_utc = newest_utc

        return all_events
