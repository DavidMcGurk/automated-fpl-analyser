from functools import cache
from concurrent.futures import ThreadPoolExecutor

from httpx import Client
from src.models.post_prediction import Position
from src.models.errors import SeasonEndedError

API_URL = "https://fantasy.premierleague.com/api"
MAX_WORKERS = 16


class ApiClient:
    def __init__(self) -> None:
        self.client = Client()

    @cache
    def get_general_info(self) -> dict:
        return self.client.get(url=f"{API_URL}/bootstrap-static/").json()

    def get_current_gw(self) -> int:
        random_player_info = self.get_player_info(1)
        remaining_fixtures = random_player_info["fixtures"]

        if not remaining_fixtures:
            raise SeasonEndedError("No more fixtures to be played!")

        return sorted(remaining_fixtures, key=lambda x: x["event"])[0]["event"]

    def get_number_of_players(self) -> int:
        general_info = self.get_general_info()
        return len(general_info["elements"])

    def get_player_info(self, player_id: int) -> dict:
        return self.client.get(f"{API_URL}/element-summary/{player_id}/").json()

    def get_player_info_batch(self, player_ids: list[int]) -> dict[int, dict]:
        """Fetch player info for multiple IDs concurrently.

        Returns a dict mapping player_id -> player info dict.
        """
        results: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(self.get_player_info, pid): pid for pid in player_ids}
            for future in futures:
                pid = futures[future]
                results[pid] = future.result()
        return results

    def get_user_summary(self, user_id: int) -> dict:
        return self.client.get(f"{API_URL}/entry/{user_id}/").json()

    def get_user_picks(self, user_id: int, gameweek: int) -> dict:
        return self.client.get(f"{API_URL}/entry/{user_id}/event/{gameweek}/picks/").json()

    def get_user_transfers(self, user_id: int) -> list[dict]:
        """Fetch the full transfer history for a user."""
        return self.client.get(f"{API_URL}/entry/{user_id}/transfers/").json()

    def get_fixtures(self) -> list[dict]:
        """Fetch all fixtures from the FPL API."""
        return self.client.get(f"{API_URL}/fixtures/").json()

    def get_player_position(self, player_element: dict) -> Position:
        element_type = player_element["element_type"]
        return Position(element_type)
