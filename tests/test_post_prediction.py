"""Tests for post-prediction models."""

from src.models.post_prediction import Position, PostPlayer


class TestPosition:
    def test_position_values(self):
        assert Position.GOALKEEPER.value == 1
        assert Position.DEFENDER.value == 2
        assert Position.MIDFIELDER.value == 3
        assert Position.ATTACKER.value == 4

    def test_position_from_int(self):
        assert Position(1) == Position.GOALKEEPER
        assert Position(4) == Position.ATTACKER


class TestPostPlayer:
    def test_basic_post_player(self):
        p = PostPlayer(
            player_id=1,
            position=Position.GOALKEEPER,
            team=1,
            xp_series=[1.5, 2.0, 3.5],
            current_price=5.0,
        )
        assert p.player_id == 1
        assert p.xp_series == [1.5, 2.0, 3.5]
        assert p.current_price == 5.0

    def test_xp_series_floats(self):
        """xp_series should accept float values."""
        p = PostPlayer(
            player_id=2,
            position=Position.ATTACKER,
            team=5,
            xp_series=[0.1, 0.5, 1.25],
            current_price=10.5,
        )
        assert all(isinstance(x, float) for x in p.xp_series)
