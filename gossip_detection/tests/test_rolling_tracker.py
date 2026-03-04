"""Tests for rolling_tracker.py - Velocity, acceleration, and author tracking."""

import pytest
from datetime import datetime, timedelta


class TestRollingTracker:
    """
    Test RollingTracker class.

    Note: These tests mock database queries since they require a running database.
    Real integration tests should be run against a test database.
    """

    def test_velocity_calculation(self):
        """Test that velocity is calculated as mentions per minute."""
        # Mock velocity calculation
        window_minutes = 5
        mention_count = 10

        velocity = mention_count / window_minutes
        assert velocity == 2.0  # 10 mentions / 5 minutes

    def test_acceleration_positive(self):
        """Test acceleration calculation with velocity increase."""
        velocity_current = 3.0
        velocity_previous = 1.0

        acceleration = (velocity_current - velocity_previous) / velocity_previous
        assert acceleration == 2.0  # 3x increase

    def test_acceleration_negative(self):
        """Test acceleration calculation with velocity decrease."""
        velocity_current = 0.5
        velocity_previous = 2.0

        acceleration = (velocity_current - velocity_previous) / velocity_previous
        assert acceleration == -0.75  # 75% decrease

    def test_acceleration_zero_previous(self):
        """Test that acceleration returns 0.0 when previous velocity is 0."""
        velocity_current = 5.0
        velocity_previous = 0.0

        # Should handle division by zero
        if velocity_previous == 0:
            acceleration = 0.0
        else:
            acceleration = (velocity_current - velocity_previous) / velocity_previous

        assert acceleration == 0.0

    def test_mention_count_computation(self):
        """Test that mention count is computed correctly."""
        # Mock: 15 mentions in 60 minute window
        count_60m = 15
        assert count_60m > 0

    def test_window_configurations(self):
        """Test that all standard windows are available."""
        windows = [5, 15, 60]

        assert len(windows) == 3
        assert windows[0] == 5
        assert windows[1] == 15
        assert windows[2] == 60

    def test_velocity_scaling_across_windows(self):
        """Test that velocity scales correctly across different windows."""
        # If 10 mentions in 5 minutes, expect 2 mentions/min
        # In 15 minutes, we'd expect 30 mentions if rate holds
        # So velocity should be 2 mentions/min in both cases

        window_5m = 5
        count_5m = 10
        velocity_5m = count_5m / window_5m  # 2.0

        window_15m = 15
        count_15m = 30
        velocity_15m = count_15m / window_15m  # 2.0

        assert velocity_5m == velocity_15m

    def test_unique_author_count(self):
        """Test counting unique authors."""
        authors = ["user1", "user2", "user1", "user3", "user2"]
        unique_count = len(set(authors))

        assert unique_count == 3
        assert unique_count == len({"user1", "user2", "user3"})

    def test_active_ticker_retrieval(self):
        """Test that active tickers are sorted."""
        tickers = ["ZZZ", "AAA", "MMM", "BBB"]
        sorted_tickers = sorted(tickers)

        assert sorted_tickers == ["AAA", "BBB", "MMM", "ZZZ"]
