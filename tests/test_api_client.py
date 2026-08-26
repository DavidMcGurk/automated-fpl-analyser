"""Tests for ApiClient."""

from unittest.mock import MagicMock, patch

from src.api_client.client import ApiClient
from src.models.post_prediction import Position
from src.models.errors import SeasonEndedError


class TestApiClient:
    @patch("src.api_client.client.Client")
    def test_get_general_info(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get.return_value.json.return_value = {"elements": [{"id": 1}]}

        client = ApiClient()
        result = client.get_general_info()
        assert "elements" in result
        assert len(result["elements"]) == 1

    @patch("src.api_client.client.Client")
    def test_get_number_of_players(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get.return_value.json.return_value = {"elements": [{"id": 1}, {"id": 2}]}

        client = ApiClient()
        # get_general_info is cached, so we need to clear it
        client.get_general_info.cache_clear()
        assert client.get_number_of_players() == 2

    @patch("src.api_client.client.Client")
    def test_get_player_info(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get.return_value.json.return_value = {"fixtures": [], "history": []}

        client = ApiClient()
        result = client.get_player_info(42)
        assert "fixtures" in result
        assert "history" in result

    @patch("src.api_client.client.Client")
    def test_get_user_summary(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get.return_value.json.return_value = {"id": 12345, "name": "Test User"}

        client = ApiClient()
        result = client.get_user_summary(12345)
        assert result["id"] == 12345

    @patch("src.api_client.client.Client")
    def test_get_user_picks(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get.return_value.json.return_value = {"picks": [{"element": 1, "position": 1}]}

        client = ApiClient()
        result = client.get_user_picks(12345, 1)
        assert len(result["picks"]) == 1
        assert result["picks"][0]["element"] == 1

    @patch("src.api_client.client.Client")
    def test_get_current_gw(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get.return_value.json.return_value = {
            "fixtures": [
                {"event": 5, "is_home": True},
                {"event": 3, "is_home": False},
            ]
        }

        client = ApiClient()
        # get_player_info is called inside get_current_gw
        gw = client.get_current_gw()
        assert gw == 3  # Should return the earliest event

    @patch("src.api_client.client.Client")
    def test_get_current_gw_season_ended(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get.return_value.json.return_value = {"fixtures": []}

        client = ApiClient()
        try:
            client.get_current_gw()
            assert False, "Should have raised SeasonEndedError"
        except SeasonEndedError:
            pass

    @patch("src.api_client.client.Client")
    def test_get_player_position(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        client = ApiClient()
        assert client.get_player_position({"element_type": 1}) == Position.GOALKEEPER
        assert client.get_player_position({"element_type": 2}) == Position.DEFENDER
        assert client.get_player_position({"element_type": 3}) == Position.MIDFIELDER
        assert client.get_player_position({"element_type": 4}) == Position.ATTACKER

    @patch("src.api_client.client.Client")
    def test_get_player_info_batch(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get.return_value.json.return_value = {"fixtures": [], "history": []}

        client = ApiClient()
        result = client.get_player_info_batch([1, 2, 3])
        assert len(result) == 3
        assert set(result.keys()) == {1, 2, 3}
        for pid in result:
            assert "fixtures" in result[pid]
            assert "history" in result[pid]

    @patch("src.api_client.client.Client")
    def test_get_user_transfers(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get.return_value.json.return_value = [
            {"element_in": 10, "element_out": 5, "event": 1},
        ]

        client = ApiClient()
        result = client.get_user_transfers(12345)
        assert len(result) == 1
        assert result[0]["event"] == 1

    @patch("src.api_client.client.Client")
    def test_get_fixtures(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get.return_value.json.return_value = [
            {"id": 1, "event": 1, "finished": True},
        ]

        client = ApiClient()
        result = client.get_fixtures()
        assert len(result) == 1
        assert result[0]["finished"] is True
