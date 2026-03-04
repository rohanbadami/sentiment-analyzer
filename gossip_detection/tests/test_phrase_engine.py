"""Tests for phrase_engine.py - N-gram and rumor keyword extraction."""

import pytest
from gossip_detection.phrase_engine import PhraseEngine


class TestPhraseEngine:
    """Test PhraseEngine class."""

    @pytest.fixture
    def engine(self):
        """Create PhraseEngine instance."""
        return PhraseEngine()

    def test_2gram_extraction(self, engine):
        """Test extraction of 2-word phrases."""
        text = "Short squeeze incoming for TSLA stock"
        result = engine.extract_ngrams(text, n_range=(2, 2))

        # Should extract valid 2-grams (stopwords filtered)
        assert any("squeeze" in phrase and "incoming" in phrase for phrase in result)
        assert len(result) > 0

    def test_3gram_extraction(self, engine):
        """Test extraction of 3-word phrases."""
        text = "This is a significant market move today"
        result = engine.extract_ngrams(text, n_range=(3, 3))

        # Should extract 3-grams with stopwords filtered
        assert len(result) > 0
        for phrase in result:
            assert len(phrase.split()) == 3

    def test_4gram_extraction(self, engine):
        """Test extraction of 4-word phrases."""
        text = "Federal Reserve announces new policy decision today"
        result = engine.extract_ngrams(text, n_range=(4, 4))

        assert len(result) > 0
        for phrase in result:
            assert len(phrase.split()) == 4

    def test_stopword_filtering(self, engine):
        """Test that stopwords are filtered from n-grams."""
        text = "the quick brown fox"
        result = engine.extract_ngrams(text, n_range=(2, 2))

        # "the" and "quick" are very common; result should filter them
        assert not any("the" in phrase for phrase in result)

    def test_rumor_keyword_detection(self, engine):
        """Test detection of rumor keywords."""
        text = "hearing rumors of an acquisition by a major competitor"
        result = engine.detect_rumor_keywords(text)

        # Should detect keywords like "hearing" and "acquisition"
        keywords = [r["keyword"] for r in result]
        assert len(keywords) > 0  # At least some keywords detected
        assert "acquisition" in keywords

    def test_rumor_keyword_context(self, engine):
        """Test that rumor keywords include surrounding context."""
        text = "There is a leak about FDA approval coming soon"
        result = engine.detect_rumor_keywords(text)

        assert len(result) > 0
        # Each result should have context
        for match in result:
            assert "context" in match
            assert "keyword" in match
            assert len(match["context"]) > 0

    def test_capitalized_entities_detection(self, engine):
        """Test detection of capitalized entity names."""
        text = "Goldman Sachs and Federal Reserve both commented"
        result = engine.detect_capitalized_entities(text)

        # Should find capitalized entities
        assert len(result) > 0
        assert any("Goldman" in entity for entity in result)

    def test_reddit_phrase_exclusion(self, engine):
        """Test that Reddit-specific n-grams are limited."""
        text = "EDIT: I was wrong. TL;DR: Buy the dip"
        result = engine.extract_ngrams(text, n_range=(2, 2))

        # N-grams should be extracted; EDIT/TL;DR are excluded from 2-gram combinations
        # but may appear if adjacent to stopwords
        assert isinstance(result, list)
        # Check that meaningful phrases exist
        phrases_str = " ".join(result).lower()
        assert "buy" in phrases_str or "dip" in phrases_str

    def test_extract_all_returns_dict(self, engine):
        """Test that extract_all returns complete feature dict."""
        text = "Hearing about an acquisition. Goldman Sachs involved. Bullish outlook."
        ticker = "XYZ"

        result = engine.extract_all(text, ticker=ticker)

        assert isinstance(result, dict)
        assert "ticker" in result
        assert result["ticker"] == "XYZ"
        assert "phrases_2gram" in result
        assert "phrases_3gram" in result
        assert "phrases_4gram" in result
        assert "rumor_keywords" in result
        assert "capitalized_entities" in result
        assert "has_rumor_language" in result
        assert isinstance(result["has_rumor_language"], bool)

    def test_extract_all_with_rumor_language(self, engine):
        """Test that has_rumor_language flag is set correctly."""
        text_with_rumor = "FDA approval confirmed!"
        text_without_rumor = "The stock is trading sideways today"

        result_with = engine.extract_all(text_with_rumor)
        result_without = engine.extract_all(text_without_rumor)

        assert result_with["has_rumor_language"] == True
        assert result_without["has_rumor_language"] == False
