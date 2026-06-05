import json
from pathlib import Path

from src.api_client.client import ApiClient
from src.models.pre_processing import PlayerAttributes, RawPlayer
from src.models.post_prediction import Position
from src.optimisation.player_feature_transformer import PlayerFeatureTransformer

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data/player_features"

GK_PATH = DATA_DIR / "goalkeepers.jsonl"
DEF_PATH = DATA_DIR / "defenders.jsonl"
MID_PATH = DATA_DIR / "midfielders.jsonl"
ATT_PATH = DATA_DIR / "attackers.jsonl"


class Predictor:
    def __init__(self) -> None:
        self.api_client = ApiClient()

    def load_player_data(self) -> None:
        num_players = self.api_client.get_number_of_players()
        general_info = self.api_client.get_general_info()

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        files = {
            Position.GOALKEEPER: GK_PATH,
            Position.DEFENDER: DEF_PATH,
            Position.MIDFIELDER: MID_PATH,
            Position.ATTACKER: ATT_PATH,
        }

        handles = {pos: path.open("w") for pos, path in files.items()}

        try:
            for i in range(1, num_players + 1):

                print(f"\nLoading player {i}")

                player_element = general_info["elements"][i - 1]

                position = self.api_client.get_player_position(player_element=player_element)

                data = self.api_client.get_player_info(player_id=i)

                attributes = PlayerAttributes(**player_element)

                raw_player = RawPlayer(player_id=i, position=position, attributes=attributes, **data)

                print(
                    f"Parsed Player -> "
                    f"id={raw_player.player_id}, "
                    f"name={raw_player.attributes.web_name}, "
                    f"position={raw_player.position}"
                )

                features = PlayerFeatureTransformer.transform(raw_player)

                handle = handles[position]

                handle.write(json.dumps(features.model_dump()) + "\n")

        finally:
            for h in handles.values():
                h.close()

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
