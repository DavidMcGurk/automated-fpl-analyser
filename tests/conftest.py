"""Shared test fixtures and helpers."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

BASE_DIR = Path(__file__).resolve().parents[1]


def _make_mock_store():
    """Create a MagicMock that behaves like a MongoStore without connecting."""
    store = MagicMock()
    store.list_seasons.return_value = []
    store.load_training_examples.return_value = []
    store.load_player_features.return_value = []
    store.load_all_predictions.return_value = {}
    store.upsert_training_examples.return_value = 0
    store.upsert_player_features.return_value = 0
    store.upsert_predictions.return_value = 0
    return store


def _make_player_attributes(**overrides):
    """Create a minimal PlayerAttributes dict that passes Pydantic validation."""
    defaults = {
        "can_transact": True,
        "can_select": True,
        "chance_of_playing_next_round": 100,
        "chance_of_playing_this_round": 100,
        "code": 12345,
        "cost_change_event": 0,
        "cost_change_event_fall": 0,
        "cost_change_start": 0,
        "cost_change_start_fall": 0,
        "price_change_percent": "0.0",
        "dreamteam_count": 0,
        "ep_next": None,
        "ep_this": None,
        "event_points": 0,
        "first_name": "Test",
        "second_name": "Player",
        "web_name": "Player",
        "known_name": "Test Player",
        "form": "0.0",
        "in_dreamteam": False,
        "now_cost": 5.0,
        "points_per_game": "0.0",
        "removed": False,
        "selected_by_percent": "0.0",
        "special": False,
        "team": 1,
        "total_points": 0,
        "transfers_in": 0,
        "transfers_in_event": 0,
        "transfers_out": 0,
        "transfers_out_event": 0,
        "value_form": "0.0",
        "value_season": "0.0",
        "minutes": 0,
        "goals_scored": 0,
        "assists": 0,
        "clean_sheets": 0,
        "goals_conceded": 0,
        "own_goals": 0,
        "penalties_saved": 0,
        "penalties_missed": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "saves": 0,
        "bonus": 0,
        "bps": 0,
        "influence": "0.0",
        "creativity": "0.0",
        "threat": "0.0",
        "ict_index": "0.0",
        "clearances_blocks_interceptions": 0,
        "recoveries": 0,
        "tackles": 0,
        "defensive_contribution": 0,
        "starts": 0,
        "expected_goals": "0.0",
        "expected_assists": "0.0",
        "expected_goal_involvements": "0.0",
        "expected_goals_conceded": "0.0",
        "corners_and_indirect_freekicks_order": None,
        "direct_freekicks_order": None,
        "penalties_order": None,
        "influence_rank": 1,
        "influence_rank_type": 1,
        "creativity_rank": 1,
        "creativity_rank_type": 1,
        "threat_rank": 1,
        "threat_rank_type": 1,
        "ict_index_rank": 1,
        "ict_index_rank_type": 1,
        "expected_goals_per_90": 0.0,
        "saves_per_90": 0.0,
        "expected_assists_per_90": 0.0,
        "expected_goal_involvements_per_90": 0.0,
        "expected_goals_conceded_per_90": 0.0,
        "goals_conceded_per_90": 0.0,
        "now_cost_rank": 1,
        "now_cost_rank_type": 1,
        "form_rank": 1,
        "form_rank_type": 1,
        "points_per_game_rank": 1,
        "points_per_game_rank_type": 1,
        "selected_rank": 1,
        "selected_rank_type": 1,
        "starts_per_90": 0.0,
        "clean_sheets_per_90": 0.0,
        "defensive_contribution_per_90": 0.0,
    }
    defaults.update(overrides)
    return defaults


def _make_history(**overrides):
    """Create a minimal History dict."""
    defaults = {
        "total_points": 5,
        "was_home": True,
        "minutes": 90,
        "goals_scored": 0,
        "assists": 0,
        "clean_sheets": 0,
        "goals_conceded": 0,
        "own_goals": 0,
        "penalties_saved": 0,
        "penalties_missed": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "saves": 0,
        "bonus": 0,
        "bps": 0,
        "influence": "0.0",
        "creativity": "0.0",
        "threat": "0.0",
        "ict_index": "0.0",
        "clearances_blocks_interceptions": 0,
        "recoveries": 0,
        "tackles": 0,
        "defensive_contribution": 0,
        "starts": 1,
        "expected_goals": "0.0",
        "expected_assists": "0.0",
        "expected_goal_involvements": "0.0",
        "expected_goals_conceded": "0.0",
        "value": 50,
        "transfers_balance": 0,
        "selected": 0,
        "transfers_in": 0,
        "transfers_out": 0,
    }
    defaults.update(overrides)
    return defaults


def _make_season_history(**overrides):
    """Create a minimal SeasonHistory dict."""
    defaults = {
        "season_name": "2024_25",
        "element_code": 12345,
        "start_cost": 45,
        "end_cost": 50,
        "total_points": 100,
        "minutes": 2000,
        "goals_scored": 5,
        "assists": 3,
        "clean_sheets": 8,
        "goals_conceded": 20,
        "own_goals": 0,
        "penalties_saved": 0,
        "penalties_missed": 0,
        "yellow_cards": 2,
        "red_cards": 0,
        "saves": 0,
        "bonus": 10,
        "bps": 200,
        "influence": "50.0",
        "creativity": "30.0",
        "threat": "40.0",
        "ict_index": "35.0",
        "clearances_blocks_interceptions": 100,
        "recoveries": 50,
        "tackles": 30,
        "defensive_contribution": 20,
        "starts": 22,
        "expected_goals": "3.0",
        "expected_assists": "2.0",
        "expected_goal_involvements": "5.0",
        "expected_goals_conceded": "15.0",
    }
    defaults.update(overrides)
    return defaults


def _make_raw_player(player_id=1, position_type=1, history=None, history_past=None, **attr_overrides):
    """Build a RawPlayer-compatible dict structure for testing."""
    from src.models.pre_processing import PlayerAttributes, RawPlayer
    from src.models.post_prediction import Position

    position = Position(position_type)
    attrs = PlayerAttributes(**_make_player_attributes(**attr_overrides))
    return RawPlayer(
        player_id=player_id,
        position=position,
        attributes=attrs,
        fixtures=[],
        history=history or [],
        history_past=history_past or [],
    )


@pytest.fixture
def make_player_attributes():
    return _make_player_attributes


@pytest.fixture
def make_history():
    return _make_history


@pytest.fixture
def make_season_history():
    return _make_season_history


@pytest.fixture
def make_raw_player():
    return _make_raw_player


@pytest.fixture
def mock_api_client():
    """A mocked ApiClient that doesn't make real HTTP calls."""
    client = MagicMock()
    return client


@pytest.fixture
def mock_store():
    """A mocked MongoStore that doesn't connect to MongoDB."""
    return _make_mock_store()
