"""Tests for historical converter helper functions."""

from src.optimisation.historical_converter import (
    _safe_float,
    _safe_int,
    _rolling_mean,
    _fixture_difficulty,
    _home_ratio,
    _build_player_gw_rows,
    _build_player_attributes,
    _build_fixtures_by_team,
)


class TestSafeFloat:
    def test_valid_float(self):
        assert _safe_float("3.5") == 3.5

    def test_int_string(self):
        assert _safe_float("5") == 5.0

    def test_none(self):
        assert _safe_float(None) == 0.0

    def test_empty_string(self):
        assert _safe_float("") == 0.0

    def test_invalid_string(self):
        assert _safe_float("abc") == 0.0

    def test_custom_default(self):
        assert _safe_float(None, default=-1.0) == -1.0


class TestSafeInt:
    def test_valid_int(self):
        assert _safe_int("42") == 42

    def test_none(self):
        assert _safe_int(None) == 0

    def test_empty_string(self):
        assert _safe_int("") == 0

    def test_invalid_string(self):
        assert _safe_int("abc") == 0

    def test_custom_default(self):
        assert _safe_int(None, default=-5) == -5


class TestRollingMean:
    def test_basic(self):
        assert _rolling_mean([1.0, 2.0, 3.0], 2) == 2.5

    def test_window_larger_than_data(self):
        assert _rolling_mean([1.0, 2.0], 5) == 1.5

    def test_empty(self):
        assert _rolling_mean([], 3) is None


class TestFixtureDifficulty:
    def test_basic(self):
        fixtures = [
            {"difficulty": 3.0, "is_home": "true"},
            {"difficulty": 4.0, "is_home": "false"},
        ]
        result = _fixture_difficulty(fixtures, 2)
        # home: 3.0 - 0.3 = 2.7, away: 4.0 + 0.3 = 4.3, mean = 3.5
        assert abs(result - 3.5) < 0.01

    def test_empty(self):
        assert _fixture_difficulty([], 3) is None

    def test_partial_window(self):
        fixtures = [{"difficulty": 2.0, "is_home": "false"}]
        result = _fixture_difficulty(fixtures, 5)
        assert abs(result - 2.3) < 0.01  # 2.0 + 0.3


class TestHomeRatio:
    def test_all_home(self):
        fixtures = [{"is_home": "true"}, {"is_home": "true"}]
        assert _home_ratio(fixtures, 2) == 1.0

    def test_all_away(self):
        fixtures = [{"is_home": "false"}, {"is_home": "false"}]
        assert _home_ratio(fixtures, 2) == 0.0

    def test_mixed(self):
        fixtures = [{"is_home": "true"}, {"is_home": "false"}]
        assert _home_ratio(fixtures, 2) == 0.5

    def test_empty(self):
        assert _home_ratio([], 3) is None


class TestBuildPlayerGwRows:
    def test_grouping(self):
        merged_gw = [
            {"element": 1, "round": 2, "total_points": 5},
            {"element": 2, "round": 1, "total_points": 3},
            {"element": 1, "round": 1, "total_points": 4},
        ]
        result = _build_player_gw_rows(merged_gw)
        assert 1 in result
        assert 2 in result
        # Player 1 should be sorted by gameweek
        assert result[1][0]["round"] == 1
        assert result[1][1]["round"] == 2

    def test_invalid_element(self):
        merged_gw = [{"element": "", "round": 1}]
        result = _build_player_gw_rows(merged_gw)
        assert len(result) == 0


class TestBuildPlayerAttributes:
    def test_indexing(self):
        players_raw = [
            {"id": 1, "web_name": "Player1"},
            {"id": 2, "web_name": "Player2"},
        ]
        result = _build_player_attributes(players_raw)
        assert result[1]["web_name"] == "Player1"
        assert result[2]["web_name"] == "Player2"

    def test_invalid_id_skipped(self):
        players_raw = [{"id": "", "web_name": "Bad"}, {"id": 5, "web_name": "Good"}]
        result = _build_player_attributes(players_raw)
        assert 5 in result
        assert -1 not in result


class TestBuildFixturesByTeam:
    def test_basic(self):
        fixtures = [
            {"team_h": 1, "team_a": 2, "event": 1, "team_h_difficulty": 2, "team_a_difficulty": 4},
            {"team_h": 2, "team_a": 1, "event": 2, "team_h_difficulty": 3, "team_a_difficulty": 3},
        ]
        result = _build_fixtures_by_team(fixtures)
        assert 1 in result
        assert 2 in result
        # Team 1: GW1 home (diff 2), GW2 away (diff 3)
        assert len(result[1]) == 2
        assert result[1][0]["event"] == 1
        assert result[1][0]["is_home"] == "true"
        assert result[1][1]["is_home"] == "false"
