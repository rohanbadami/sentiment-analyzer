"""
xpoz_ingestion.py - Reddit poller using Xpoz MCP API.

Replaces Pullpush.io (dead after May 2025) with Xpoz MCP, which provides
live Reddit data. Xpoz uses an async polling pattern: tool calls return an
operationId that must be polled via checkOperationStatus until complete.
Results are returned as a compact YAML-like text format, not raw JSON.
"""

import csv
import io
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from gossip_detection.config import (
    CHECKPOINT_FILE,
    MAX_WORKERS,
    XPOZ_API_KEY,
    XPOZ_KEYWORDS,
    XPOZ_MCP_URL,
)
from gossip_detection.event_schema import RedditEvent

logger = logging.getLogger(__name__)


class XpozPoller:
    """
    Fetches Reddit posts and comments from Xpoz MCP API.

    Drop-in replacement for RedditPoller. Same checkpoint file, same poll_all()
    interface, same ThreadPoolExecutor pattern.

    Xpoz async pattern:
      1. POST to /mcp with a tool call → get back operationId (status: running)
      2. Poll checkOperationStatus with operationId every 5s until status != running
      3. Parse compact text-format results into RedditEvent objects
    """

    def __init__(self, subreddits: list[str]):
        self.subreddits = subreddits

        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.headers.update({
            "Authorization": f"Bearer {XPOZ_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        })

        self.last_poll_utc = self._load_checkpoint()

    # -------------------------------------------------------------------------
    # Checkpoint helpers
    # -------------------------------------------------------------------------

    def _load_checkpoint(self) -> int:
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

        default_ts = int(time.time()) - 3600
        logger.info(f"No checkpoint found. Defaulting to 1 hour ago: {default_ts}")
        return default_ts

    def _save_checkpoint(self, utc_timestamp: int):
        checkpoint_path = Path(CHECKPOINT_FILE)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(checkpoint_path, "w") as f:
                json.dump({"last_poll_utc": utc_timestamp}, f)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    # -------------------------------------------------------------------------
    # Xpoz MCP async call machinery
    # -------------------------------------------------------------------------

    def _extract_sse_text(self, raw: str) -> tuple[str, str]:
        """
        Extract (text_content, operation_id) from an SSE response body.

        Returns (text, op_id). Either may be empty string if not found.
        """
        for line in raw.splitlines():
            if not line.startswith("data:"):
                continue
            try:
                msg = json.loads(line[5:].strip())
                result = msg.get("result", {})
                op_id = result.get("_meta", {}).get("progressToken", "")
                content = result.get("content", [])
                for block in content:
                    if block.get("type") == "text":
                        return block["text"], op_id
            except Exception as e:
                logger.debug(f"SSE parse error: {e} | line[:80]={line[:80]}")
        return "", ""

    def _post_tool_raw(self, tool_name: str, arguments: dict, call_id: int = 1) -> str:
        """POST a tool call and return the full response body as text (streaming-safe)."""
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
            "id": call_id,
        }
        resp = self.session.post(XPOZ_MCP_URL, json=payload, timeout=60, stream=True)
        resp.raise_for_status()
        return resp.content.decode("utf-8", errors="replace")

    def _call_tool(self, tool_name: str, arguments: dict, max_wait: int = 120) -> str:
        """
        Call an Xpoz MCP tool and wait for the async result.

        Returns the raw text content block. Handles both:
        - Synchronous (fast) responses: results returned immediately in the SSE body
        - Async responses: operationId returned, polled via checkOperationStatus
        """
        raw = self._post_tool_raw(tool_name, arguments)
        text, op_id = self._extract_sse_text(raw)

        if not text and not op_id:
            logger.warning(f"Xpoz [{tool_name}]: unparseable response. raw[:200]={raw[:200]}")
            return ""

        logger.debug(f"Xpoz [{tool_name}] initiated. op_id={op_id or '(sync)'}")

        if not op_id:
            # Fast / synchronous response
            return text

        # Async: poll until done
        deadline = time.time() + max_wait
        while time.time() < deadline:
            time.sleep(5)
            try:
                poll_raw = self._post_tool_raw("checkOperationStatus", {"operationId": op_id}, call_id=2)
                poll_text, _ = self._extract_sse_text(poll_raw)
                logger.debug(f"Xpoz poll [{op_id[:40]}]: {poll_text[:120]}")

                if poll_text and "status: running" not in poll_text:
                    return poll_text
            except Exception as e:
                logger.warning(f"Xpoz poll error for {op_id}: {e}")

        logger.error(f"Xpoz [{tool_name}] timed out after {max_wait}s")
        return ""

    # -------------------------------------------------------------------------
    # Response text parsers
    # -------------------------------------------------------------------------

    def _parse_results(self, text: str) -> list[dict]:
        """
        Parse Xpoz compact text results into a list of dicts.

        Handles the `results[N]{col1,col2,...}:` format returned by both
        fast (posts) and paged (comments) responses.
        """
        lines = text.splitlines()
        columns: list[str] = []
        rows: list[dict] = []

        for i, line in enumerate(lines):
            # Match: results[N]{col1,col2,...}:
            m = re.search(r'results\[\d+\]\{([^}]+)\}:', line)
            if not m:
                continue

            columns = [c.strip() for c in m.group(1).split(",")]
            logger.debug(f"Xpoz results columns: {columns}")

            for data_line in lines[i + 1:]:
                stripped = data_line.strip()
                if not stripped:
                    continue
                # Stop at next YAML key (e.g. "nextPageToken: ...")
                if re.match(r'^[a-zA-Z_]+:', stripped) and not stripped.startswith('"'):
                    break
                try:
                    reader = csv.reader(io.StringIO(stripped))
                    for row in reader:
                        if len(row) >= len(columns):
                            rows.append(dict(zip(columns, row[:len(columns)])))
                        elif row:
                            padded = row + [""] * (len(columns) - len(row))
                            rows.append(dict(zip(columns, padded)))
                except Exception as e:
                    logger.debug(f"CSV parse error on line '{stripped}': {e}")

            break

        return rows

    def _normalize_row(self, row: dict) -> dict:
        """Normalize Xpoz field names to RedditEvent-compatible names."""
        return {
            "id": row.get("id", ""),
            "author": row.get("authorUsername") or row.get("author", ""),
            # posts use subredditName; comments use postSubredditName
            "subreddit": row.get("subredditName") or row.get("postSubredditName") or row.get("subreddit", ""),
            "title": row.get("title"),
            "created_utc": row.get("createdAtDate") or row.get("created_utc") or row.get("created"),
            "selftext": row.get("selftext") or row.get("body") or row.get("text") or "",
            "score": self._to_int(row.get("score") or row.get("upvotes") or row.get("ups")),
            "num_comments": self._to_int(row.get("commentsCount") or row.get("num_comments") or row.get("numComments")),
            "permalink": row.get("permalink") or row.get("url") or "",
        }

    @staticmethod
    def _to_int(val) -> int:
        try:
            return int(val or 0)
        except (ValueError, TypeError):
            return 0

    # -------------------------------------------------------------------------
    # Fetch methods
    # -------------------------------------------------------------------------

    def fetch_posts_by_keywords(self) -> list[RedditEvent]:
        """
        Fetch up to 300 posts matching financial keywords (1 API call, synchronous).

        Uses responseType=fast which returns immediately without async polling.
        sort=new + time=day ensures we get the freshest content each cycle.
        """
        text = self._call_tool(
            "getRedditPostsByKeywords",
            {
                "query": XPOZ_KEYWORDS,
                "sort": "new",
                "time": "day",
                "responseType": "fast",
                # Exclude selftext — multiline/quoted content breaks compact CSV format.
                # Title is sufficient for ticker extraction; full text via permalink if needed.
                "fields": ["id", "title", "authorUsername", "subredditName",
                           "score", "commentsCount", "permalink", "createdAtDate"],
            },
        )
        if not text:
            logger.warning("Xpoz keyword posts: empty response")
            return []

        rows = self._parse_results(text)
        events = []
        for row in rows:
            normalized = self._normalize_row(row)
            event = RedditEvent.from_xpoz_json(normalized, post_type="post")
            if event:
                events.append(event)

        logger.info(f"Xpoz keyword posts: raw={len(rows)}, parsed={len(events)}")
        return events

    def fetch_comments_by_keywords(self) -> list[RedditEvent]:
        """
        Fetch up to 100 comments matching financial keywords (1 API call, async).

        Comments don't support responseType=fast; uses async polling automatically.
        """
        text = self._call_tool(
            "getRedditCommentsByKeywords",
            {
                "query": XPOZ_KEYWORDS,
                "fields": ["id", "body", "authorUsername", "postSubredditName",
                           "score", "createdAtDate", "permalink"],
            },
        )
        if not text:
            logger.warning("Xpoz keyword comments: empty response")
            return []

        rows = self._parse_results(text)
        events = []
        for row in rows:
            normalized = self._normalize_row(row)
            event = RedditEvent.from_xpoz_json(normalized, post_type="comment")
            if event:
                events.append(event)

        logger.info(f"Xpoz keyword comments: raw={len(rows)}, parsed={len(events)}")
        return events

    # -------------------------------------------------------------------------
    # Main poll interface (same as RedditPoller)
    # -------------------------------------------------------------------------

    def poll_all(self, since_utc: int) -> list[RedditEvent]:
        """
        Fetch posts + comments via 2 keyword searches (covers all subreddits).

        2 calls/cycle × 480 cycles/month × 100 results = 96K results/month,
        safely under the 100K Xpoz free-tier limit at 90-minute intervals.
        """
        all_events: list[RedditEvent] = []
        poll_start = time.time()
        logger.info("Starting Xpoz poll cycle (keyword mode: 2 calls total)...")

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(self.fetch_posts_by_keywords),
                executor.submit(self.fetch_comments_by_keywords),
            ]
            for future in as_completed(futures):
                try:
                    all_events.extend(future.result())
                except Exception as e:
                    logger.error(f"Xpoz fetch error: {e}")

        # Deduplicate by reddit_id
        seen: set[str] = set()
        deduped: list[RedditEvent] = []
        for e in all_events:
            if e.reddit_id not in seen:
                seen.add(e.reddit_id)
                deduped.append(e)

        # Sort ascending and filter to new-only
        deduped.sort(key=lambda e: e.created_utc or datetime.utcnow())
        new_events = [
            e for e in deduped
            if e.created_utc and e.created_utc.timestamp() > since_utc
        ]

        cycle_ms = (time.time() - poll_start) * 1000
        logger.info(
            f"Xpoz poll complete: {len(new_events)} new events "
            f"(of {len(deduped)} total), cycle_time={cycle_ms:.0f}ms"
        )

        if new_events:
            newest_utc = int(new_events[-1].created_utc.timestamp())
            self._save_checkpoint(newest_utc)
            self.last_poll_utc = newest_utc

        return new_events
