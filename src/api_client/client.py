from httpx import Client

from src.api_client.models import SeasonEndedError

API_URL = 'https://fantasy.premierleague.com/api'


class ApiClient:
    def __init__(self) -> None:
        self.client = Client()

    def get_general_info(self) -> dict:
        return self.client.get(url=f'{API_URL}/bootstrap-static/').json()

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
        return self.client.get(f'{API_URL}/element-summary/{player_id}/').json()

    def get_user_summary(self, user_id: int) -> dict:
        return self.client.get(f'{API_URL}/api/entry/{user_id}/').json()
