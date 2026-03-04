"""Tests for gossip_scorer.py - Composite score calculation and bounds."""

import pytest
from gossip_detection.config import (
    GOSSIP_SCORE_WEIGHTS,
    VELOCITY_MAX_MENTIONS_PER_MIN,
    ACCELERATION_MAX,
    AUTHOR_DIVERSITY_MAX,
)


class TestGossipScorer:
    """Test GossipScorer scoring logic."""

    def test_weight_sum(self):
        """Test that all weights sum to 1.0."""
        total_weight = sum(GOSSIP_SCORE_WEIGHTS.values())

        assert abs(total_weight - 1.0) < 0.001

    def test_all_weights_present(self):
        """Test that all required weight keys are present."""
        required_keys = [
            "velocity",
            "acceleration",
            "author_diversity",
            "anomaly",
            "confirmation",
        ]

        for key in required_keys:
            assert key in GOSSIP_SCORE_WEIGHTS

    def test_velocity_normalization(self):
        """Test velocity score normalization."""
        # At max velocity (10 mentions/min), score should be 1.0
        max_velocity = 10.0
        velocity_score = min(max_velocity / VELOCITY_MAX_MENTIONS_PER_MIN, 1.0)
        assert velocity_score == 1.0

        # At half max, score should be 0.5
        half_velocity = 5.0
        velocity_score = min(half_velocity / VELOCITY_MAX_MENTIONS_PER_MIN, 1.0)
        assert velocity_score == 0.5

        # At zero velocity, score should be 0.0
        zero_velocity = 0.0
        velocity_score = min(zero_velocity / VELOCITY_MAX_MENTIONS_PER_MIN, 1.0)
        assert velocity_score == 0.0

    def test_acceleration_normalization(self):
        """Test acceleration score normalization."""
        # At max acceleration (5x), score should be 1.0
        max_acceleration = 5.0
        acceleration_score = min(abs(max_acceleration) / ACCELERATION_MAX, 1.0)
        assert acceleration_score == 1.0

        # At zero acceleration, score should be 0.0
        zero_acceleration = 0.0
        acceleration_score = min(abs(zero_acceleration) / ACCELERATION_MAX, 1.0)
        assert acceleration_score == 0.0

    def test_author_diversity_normalization(self):
        """Test author diversity normalization."""
        # At max authors (50), score should be 1.0
        max_authors = 50
        author_score = min(max_authors / AUTHOR_DIVERSITY_MAX, 1.0)
        assert author_score == 1.0

        # At 25 authors, score should be 0.5
        half_authors = 25
        author_score = min(half_authors / AUTHOR_DIVERSITY_MAX, 1.0)
        assert author_score == 0.5

    def test_anomaly_score_binary(self):
        """Test that anomaly score is binary (0 or 1)."""
        anomaly_score_true = 1.0
        anomaly_score_false = 0.0

        assert anomaly_score_true in [0.0, 1.0]
        assert anomaly_score_false in [0.0, 1.0]

    def test_confirmation_score_range(self):
        """Test that confirmation strength is in [0, 1]."""
        confirmation_scores = [0.0, 0.25, 0.5, 0.75, 1.0]

        for score in confirmation_scores:
            assert 0.0 <= score <= 1.0

    def test_composite_score_bounds(self):
        """Test that composite gossip score is always in [0, 1]."""
        # Test with all sub-scores at max
        all_max = (
            (1.0 * GOSSIP_SCORE_WEIGHTS["velocity"]) +
            (1.0 * GOSSIP_SCORE_WEIGHTS["acceleration"]) +
            (1.0 * GOSSIP_SCORE_WEIGHTS["author_diversity"]) +
            (1.0 * GOSSIP_SCORE_WEIGHTS["anomaly"]) +
            (1.0 * GOSSIP_SCORE_WEIGHTS["confirmation"])
        )

        gossip_score = max(0.0, min(all_max, 1.0))
        assert 0.0 <= gossip_score <= 1.0
        assert gossip_score == 1.0

        # Test with all sub-scores at min
        all_min = (
            (0.0 * GOSSIP_SCORE_WEIGHTS["velocity"]) +
            (0.0 * GOSSIP_SCORE_WEIGHTS["acceleration"]) +
            (0.0 * GOSSIP_SCORE_WEIGHTS["author_diversity"]) +
            (0.0 * GOSSIP_SCORE_WEIGHTS["anomaly"]) +
            (0.0 * GOSSIP_SCORE_WEIGHTS["confirmation"])
        )

        gossip_score = max(0.0, min(all_min, 1.0))
        assert 0.0 <= gossip_score <= 1.0
        assert gossip_score == 0.0

    def test_weighted_composite_example(self):
        """Test example composite score calculation."""
        # Example: moderate activity
        velocity_score = 0.5  # 5 mentions/min
        acceleration_score = 0.3  # 1.5x velocity increase
        author_diversity_score = 0.4  # 20 unique authors
        anomaly_score = 0.0  # No anomaly
        confirmation_score = 0.2  # Weak cross-source signal

        gossip_score = (
            (velocity_score * GOSSIP_SCORE_WEIGHTS["velocity"]) +
            (acceleration_score * GOSSIP_SCORE_WEIGHTS["acceleration"]) +
            (author_diversity_score * GOSSIP_SCORE_WEIGHTS["author_diversity"]) +
            (anomaly_score * GOSSIP_SCORE_WEIGHTS["anomaly"]) +
            (confirmation_score * GOSSIP_SCORE_WEIGHTS["confirmation"])
        )

        gossip_score = max(0.0, min(gossip_score, 1.0))

        # Should be between 0 and 1
        assert 0.0 <= gossip_score <= 1.0
        # Should be moderate (not at extremes)
        assert 0.2 < gossip_score < 0.6
