"""
phrase_engine.py - Phrase extraction and rumor keyword detection from Reddit text.

Extracts n-grams, detects rumor language, and identifies capitalized entities.
Follows the regex pattern approach from integrated_processor.py.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PhraseEngine:
    """
    Extracts meaningful phrases and detects rumor language from Reddit posts/comments.

    Combines n-gram extraction, keyword detection, and entity recognition.
    """

    def __init__(self):
        """Initialize phrase engine with rumor keywords and stopwords."""
        self.rumor_keywords = {
            "rumor", "leak", "leaked", "hearing", "confirmed",
            "acquisition", "partnership", "merger", "buyout", "insider",
            "fda", "approval", "investigation", "subpoena", "squeeze",
            "moon", "rocket", "manipulation", "offering", "dilution",
            "bankrupt", "delisted", "short", "cover", "pump", "dump",
            "catalyst", "earnings", "guidance", "split", "reverse split",
        }

        # Build regex patterns for each rumor keyword (word boundaries)
        self.rumor_patterns = {
            kw: re.compile(rf'\b{re.escape(kw)}\b', re.IGNORECASE)
            for kw in self.rumor_keywords
        }

        # Common stopwords for n-gram filtering
        self.stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "is", "was", "are", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "should", "could", "may", "might",
            "can", "i", "you", "he", "she", "it", "we", "they", "this", "that",
            "these", "those", "what", "which", "who", "when", "where", "why", "how",
        }

        # Reddit phrases to exclude
        self.reddit_phrases = {"edit", "tl;dr", "dd", "yolo", "op", "eta"}

    def extract_ngrams(
        self, text: str, n_range: tuple[int, int] = (2, 4)
    ) -> list[str]:
        """
        Extract n-grams from text (2-4 words by default).

        Args:
            text: Input text to extract from
            n_range: Tuple of (min_n, max_n) for gram sizes

        Returns:
            List of n-gram phrase strings
        """
        # Lowercase, remove punctuation, split on whitespace
        text_cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = text_cleaned.split()

        # Remove stopwords
        tokens = [t for t in tokens if t not in self.stopwords and len(t) > 1]

        if len(tokens) < n_range[0]:
            return []

        ngrams = []

        # Extract n-grams for each n in range
        for n in range(n_range[0], n_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i : i + n])

                # Skip if all stopwords or Reddit-specific phrases
                if not ngram or ngram in self.reddit_phrases:
                    continue

                ngrams.append(ngram)

        return ngrams

    def detect_rumor_keywords(self, text: str) -> list[dict]:
        """
        Detect rumor keywords and extract surrounding context.

        Args:
            text: Input text to search

        Returns:
            List of dicts with keyword, context, and position info
        """
        results = []

        for keyword, pattern in self.rumor_patterns.items():
            for match in pattern.finditer(text):
                # Extract surrounding context (10 words before/after)
                start_pos = match.start()
                end_pos = match.end()

                # Find word boundaries for context extraction
                context_start = max(0, start_pos - 50)  # ~10 words before
                context_end = min(len(text), end_pos + 50)  # ~10 words after

                context = text[context_start:context_end].strip()

                results.append({
                    "keyword": keyword,
                    "context": context,
                    "position": start_pos,
                })

        return results

    def detect_capitalized_entities(self, text: str) -> list[str]:
        """
        Detect sequences of 2+ capitalized words (e.g., "Goldman Sachs").

        Args:
            text: Input text to search

        Returns:
            List of entity strings
        """
        # Pattern: 2+ consecutive capitalized words
        pattern = r'(?:[A-Z][a-z]+\s+)+'
        entities = re.findall(pattern, text)

        # Clean and filter
        entities = [e.strip() for e in entities]

        # Exclude common Reddit phrases
        excluded = {"EDIT", "TL;DR", "DD", "YOLO", "OP", "ETA"}
        entities = [e for e in entities if e.upper() not in excluded]

        return entities

    def extract_all(self, text: str, ticker: str = "") -> dict:
        """
        Extract all phrase features from text.

        Args:
            text: Reddit post/comment text
            ticker: Stock ticker symbol (for metadata)

        Returns:
            Dict with all extracted features
        """
        phrases_2gram = self.extract_ngrams(text, n_range=(2, 2))
        phrases_3gram = self.extract_ngrams(text, n_range=(3, 3))
        phrases_4gram = self.extract_ngrams(text, n_range=(4, 4))

        rumor_keywords = self.detect_rumor_keywords(text)
        entities = self.detect_capitalized_entities(text)

        # Check if any rumor language detected
        has_rumor = len(rumor_keywords) > 0

        return {
            "ticker": ticker,
            "phrases_2gram": phrases_2gram,
            "phrases_3gram": phrases_3gram,
            "phrases_4gram": phrases_4gram,
            "rumor_keywords": rumor_keywords,
            "capitalized_entities": entities,
            "has_rumor_language": has_rumor,
        }
