from httpx import Client

from src.models import SeasonEndedError


class ApiClient:
    def __init__(self, user_id: int) -> None:
        self.client = Client()
        self.user_id = user_id

    def get_general_info(self) -> dict:
        return self.client.get(url='https://fantasy.premierleague.com/api/bootstrap-static/').json()

    def get_current_gw(self) -> int:
        random_player_info = self.get_player_info(1)
        remaining_fixtures = random_player_info["fixtures"]

        if not remaining_fixtures:
            raise SeasonEndedError("No more fixtures to be played!")

        return sorted(remaining_fixtures, key=lambda x: x["event"])[0]["event"]

    def get_player_info(self, player_id: int) -> dict:
        return self.client.get(f'https://fantasy.premierleague.com/api/element-summary/{player_id}/').json()

    def get_user_summary(self) -> dict:
        return self.client.get(f'https://fantasy.premierleague.com/api/entry/{self.user_id}/').json()
