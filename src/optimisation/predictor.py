import json
from pathlib import Path

from src.api_client.client import ApiClient
from src.models.pre_processing import PlayerAttributes, RawPlayer
from src.models.post_prediction import Position
from src.models.training_examples import TrainingExample
from src.optimisation.player_feature_transformer import PlayerFeatureTransformer

BASE_DIR = Path(__file__).resolve().parents[2]


FEATURE_DIR = BASE_DIR / "data/player_features"
TRAINING_BASE_DIR = BASE_DIR / "data/training"


FEATURE_PATHS = {
    Position.GOALKEEPER: FEATURE_DIR / "goalkeepers.jsonl",
    Position.DEFENDER: FEATURE_DIR / "defenders.jsonl",
    Position.MIDFIELDER: FEATURE_DIR / "midfielders.jsonl",
    Position.ATTACKER: FEATURE_DIR / "attackers.jsonl",
}


def _training_paths(season: str) -> dict[Position, Path]:
    training_dir = TRAINING_BASE_DIR / season
    return {
        Position.GOALKEEPER: training_dir / "goalkeepers.jsonl",
        Position.DEFENDER: training_dir / "defenders.jsonl",
        Position.MIDFIELDER: training_dir / "midfielders.jsonl",
        Position.ATTACKER: training_dir / "attackers.jsonl",
    }


class Predictor:

    def __init__(self) -> None:
        self.api_client = ApiClient()

    def load_player_data(self) -> None:
        num_players = self.api_client.get_number_of_players()
        general_info = self.api_client.get_general_info()

        season = self._derive_season_name(general_info)
        training_paths = _training_paths(season)
        training_dir = TRAINING_BASE_DIR / season

        FEATURE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        training_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        feature_handles = {position: path.open("w") for position, path in FEATURE_PATHS.items()}
        training_handles = {position: path.open("w") for position, path in training_paths.items()}

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

        print(f"\nTraining data written to: {training_dir}")

    @staticmethod
    def _derive_season_name(general_info: dict) -> str:
        """Derive a season folder name (e.g. '2026_27') from the API events."""
        events = general_info.get("events", [])
        if not events:
            return "current"

        last_event = events[-1]
        deadline = last_event.get("deadline_time", "")
        end_year = int(deadline[:4])

        return f"{end_year - 1}_{str(end_year)[-2:]}"

    def model_xp(self) -> None:
        """Train GP models and predict xP for all current players."""
        from src.optimisation.gp_model import GPModel

        gp = GPModel()
        gp.train()
        gp.predict()

    def optimise_team(self, user_id: int, max_transfers: int = 2) -> None:
        """Optimise a user's team by suggesting transfers that maximise xP."""
        from src.optimisation.team_optimiser import TeamOptimiser

        optimiser = TeamOptimiser()
        result = optimiser.optimise(user_id, max_transfers=max_transfers)

        print(f"\n{'=' * 60}")
        print(f"Team Optimisation Results for User {user_id}")
        print(f"{'=' * 60}")
        print(f"Current squad xP: {result.current_squad_xp}")
        print(f"Optimised squad xP: {result.optimised_squad_xp}")
        print(f"Transfers used: {result.transfers_used}")
        print(f"Point hit: {result.point_hit}")
        print(f"Net improvement: {result.optimised_squad_xp - result.current_squad_xp - result.point_hit:+.2f}")

        if result.suggestions:
            print("\nSuggested transfers:")
            for s in result.suggestions:
                print(f"  {s.player_out_name} (ID: {s.player_out}) -> " f"{s.player_in_name} (ID: {s.player_in})")
                print(f"    xP gain: {s.xP_gain:+.2f}, " f"cost change: {s.cost_change:+.2f}")
        else:
            print("\nNo beneficial transfers found.")

        print(f"{'=' * 60}")
