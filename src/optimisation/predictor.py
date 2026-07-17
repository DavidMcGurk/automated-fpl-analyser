import json
from pathlib import Path

from src.api_client.client import ApiClient
from src.models.pre_processing import PlayerAttributes, RawPlayer
from src.models.post_prediction import Position
from src.models.training_examples import TrainingExample
from src.optimisation.player_feature_transformer import PlayerFeatureTransformer

BASE_DIR = Path(__file__).resolve().parents[2]


FEATURE_DIR = BASE_DIR / "data/player_features"
TRAINING_DIR = BASE_DIR / "data/training"


FEATURE_PATHS = {
    Position.GOALKEEPER: FEATURE_DIR / "goalkeepers.jsonl",
    Position.DEFENDER: FEATURE_DIR / "defenders.jsonl",
    Position.MIDFIELDER: FEATURE_DIR / "midfielders.jsonl",
    Position.ATTACKER: FEATURE_DIR / "attackers.jsonl",
}


TRAINING_PATHS = {
    Position.GOALKEEPER: TRAINING_DIR / "goalkeepers.jsonl",
    Position.DEFENDER: TRAINING_DIR / "defenders.jsonl",
    Position.MIDFIELDER: TRAINING_DIR / "midfielders.jsonl",
    Position.ATTACKER: TRAINING_DIR / "attackers.jsonl",
}


class Predictor:

    def __init__(self) -> None:
        self.api_client = ApiClient()

    def load_player_data(self) -> None:
        num_players = self.api_client.get_number_of_players()
        general_info = self.api_client.get_general_info()

        FEATURE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        TRAINING_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        feature_handles = {position: path.open("w") for position, path in FEATURE_PATHS.items()}
        training_handles = {position: path.open("w") for position, path in TRAINING_PATHS.items()}

        try:

            for i in range(1, num_players + 1):
                print(f"\nLoading player {i}")

                player_element = general_info["elements"][i - 1]
                position = self.api_client.get_player_position(player_element)
                data = self.api_client.get_player_info(player_id=i)
                attributes = PlayerAttributes(**player_element)

                raw_player = RawPlayer(
                    player_id=i,
                    position=position,
                    attributes=attributes,
                    **data,
                )

                print(
                    f"Parsed Player -> "
                    f"id={raw_player.player_id}, "
                    f"name={raw_player.attributes.web_name}, "
                    f"position={raw_player.position}"
                )

                # Current inference features
                features = PlayerFeatureTransformer.transform(raw_player)
                feature_handles[position].write(json.dumps(features.model_dump()) + "\n")

                # Historical training rows
                examples = PlayerFeatureTransformer.transform_history(raw_player)

                for example in examples:
                    training_example = TrainingExample(**example)
                    training_handles[position].write(training_example.model_dump_json() + "\n")

        finally:
            for handle in feature_handles.values():
                handle.close()

            for handle in training_handles.values():
                handle.close()

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
