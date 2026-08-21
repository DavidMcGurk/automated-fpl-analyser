"""Tests for pre-processing Pydantic models."""

from src.models.pre_processing import PlayerAttributes, History, SeasonHistory
from src.models.post_prediction import Position


class TestPlayerAttributes:
    def test_valid_attributes(self, make_player_attributes):
        attrs = PlayerAttributes(**make_player_attributes())
        assert attrs.web_name == "Player"
        assert attrs.now_cost == 5.0

    def test_ep_this_none_allowed(self, make_player_attributes):
        """ep_this should accept None (pre-season API returns None)."""
        attrs = PlayerAttributes(**make_player_attributes(ep_this=None))
        assert attrs.ep_this is None

    def test_ep_this_string_allowed(self, make_player_attributes):
        attrs = PlayerAttributes(**make_player_attributes(ep_this="5.0"))
        assert attrs.ep_this == "5.0"

    def test_ep_next_none_allowed(self, make_player_attributes):
        attrs = PlayerAttributes(**make_player_attributes(ep_next=None))
        assert attrs.ep_next is None

    def test_optional_fields_none(self, make_player_attributes):
        """All Optional fields should accept None."""
        attrs = PlayerAttributes(
            **make_player_attributes(
                chance_of_playing_next_round=None,
                chance_of_playing_this_round=None,
                corners_and_indirect_freekicks_order=None,
                direct_freekicks_order=None,
                penalties_order=None,
            )
        )
        assert attrs.chance_of_playing_next_round is None
        assert attrs.penalties_order is None


class TestHistory:
    def test_valid_history(self, make_history):
        h = History(**make_history())
        assert h.total_points == 5
        assert h.minutes == 90

    def test_history_with_overrides(self, make_history):
        h = History(**make_history(total_points=15, goals_scored=2))
        assert h.total_points == 15
        assert h.goals_scored == 2


class TestSeasonHistory:
    def test_valid_season_history(self, make_season_history):
        sh = SeasonHistory(**make_season_history())
        assert sh.season_name == "2024_25"
        assert sh.total_points == 100


class TestRawPlayer:
    def test_raw_player_construction(self, make_raw_player):
        player = make_raw_player(player_id=10, position_type=3)
        assert player.player_id == 10
        assert player.position == Position.MIDFIELDER
        assert player.fixtures == []
        assert player.this_season_history == []
        assert player.previous_seasons_history == []

    def test_raw_player_with_history(self, make_raw_player, make_history, make_season_history):
        player = make_raw_player(
            history=[make_history(total_points=10), make_history(total_points=5)],
            history_past=[make_season_history()],
        )
        assert len(player.this_season_history) == 2
        assert len(player.previous_seasons_history) == 1
        assert player.this_season_history[0].total_points == 10
