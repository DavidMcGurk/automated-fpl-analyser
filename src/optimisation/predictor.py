import json
from pathlib import Path
from src.api_client.client import ApiClient

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data/raw_player_data.jsonl"


class Predictor:
    def __init__(self) -> None:
        self.api_client = ApiClient()
        self.output_path = DATA_DIR

    def load_player_data(self) -> None:
        num_players = self.api_client.get_number_of_players()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a") as f:
            for i in range(1, num_players + 1):
                print(f"Loading data for player {i}")

                data = self.api_client.get_player_info(player_id=i)

                record = {"player_id": i, "data": data}

                f.write(json.dumps(record) + "\n")

    def model_xp(self) -> None:
        # Produce ML model of expected points (xP) / player for next <5 fixtures, from player data and upcoming fixtures
        # Write to folder in data, describing player id, xP / player for remaining up to 5 fixtures,
        # + other rel. info (e.g. position)
        pass

    def optimise_team(self) -> None:
        # Evaluate xP of users current team
        # For each non-selected player with > avg xp, see if they can be included in team + if that improves things
        # (Subsequently) consider pairs of transfers which can most improve xP
        # Relevant constraints: position rules, player prices, club max players, budget (sale prices / player), etc.
        pass
