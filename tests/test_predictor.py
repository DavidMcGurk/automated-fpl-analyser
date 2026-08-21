"""Tests for Predictor orchestration."""

from unittest.mock import MagicMock, patch

from src.optimisation.predictor import Predictor


class TestDeriveSeasonName:
    def test_standard_season(self):
        general_info = {
            "events": [
                {"deadline_time": "2026-08-15T12:00:00Z"},
                {"deadline_time": "2027-05-30T13:30:00Z"},
            ]
        }
        assert Predictor._derive_season_name(general_info) == "2026_27"

    def test_no_events(self):
        general_info = {"events": []}
        assert Predictor._derive_season_name(general_info) == "current"

    def test_single_event(self):
        general_info = {"events": [{"deadline_time": "2025-08-10T12:00:00Z"}]}
        assert Predictor._derive_season_name(general_info) == "2024_25"

    def test_empty_dict(self):
        assert Predictor._derive_season_name({}) == "current"


class TestPredictorInit:
    @patch("src.optimisation.predictor.ApiClient")
    def test_init(self, mock_api_class):
        mock_api_class.return_value = MagicMock()
        predictor = Predictor()
        assert predictor.api_client is not None


class TestModelXp:
    @patch("src.optimisation.predictor.ApiClient")
    def test_model_xp_delegates_to_gp_model(self, mock_api_class):
        mock_api_class.return_value = MagicMock()
        predictor = Predictor()

        with patch("src.optimisation.gp_model.GPModel") as mock_gp_class:
            mock_gp = MagicMock()
            mock_gp_class.return_value = mock_gp
            predictor.model_xp()
            mock_gp.train.assert_called_once()
            mock_gp.predict.assert_called_once()


class TestOptimiseTeam:
    @patch("src.optimisation.predictor.ApiClient")
    def test_optimise_team_delegates_to_optimiser(self, mock_api_class):
        mock_api_class.return_value = MagicMock()
        predictor = Predictor()

        mock_result = MagicMock()
        mock_result.current_squad_xp = 50.0
        mock_result.optimised_squad_xp = 55.0
        mock_result.transfers_used = 1
        mock_result.point_hit = 0
        mock_result.suggestions = []

        with patch("src.optimisation.team_optimiser.TeamOptimiser") as mock_opt_class:
            mock_opt = MagicMock()
            mock_opt.optimise.return_value = mock_result
            mock_opt_class.return_value = mock_opt

            predictor.optimise_team(12345, max_transfers=2)
            mock_opt.optimise.assert_called_once_with(12345, max_transfers=2)
