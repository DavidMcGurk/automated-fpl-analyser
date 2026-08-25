import json

from src.api_client.client import ApiClient
from src.models.pre_processing import PlayerAttributes, RawPlayer
from src.models.post_prediction import Position
from src.models.training_examples import TrainingExample
from src.optimisation.player_feature_transformer import PlayerFeatureTransformer
from src.storage.mongo_client import MongoStore, POSITION_NAMES


class Predictor:

    def __init__(self, store: MongoStore | None = None) -> None:
        self.api_client = ApiClient()
        self.store = store or MongoStore()

    def load_player_data(self) -> None:
        num_players = self.api_client.get_number_of_players()
        general_info = self.api_client.get_general_info()

        season = self._derive_season_name(general_info)

        # Fetch all player details concurrently
        all_elements = general_info["elements"]
        player_ids = [e["id"] for e in all_elements]
        player_info_map = self.api_client.get_player_info_batch(player_ids)

        # Accumulate features and training examples per position
        features_by_position: dict[Position, list[dict]] = {p: [] for p in Position}
        training_by_position: dict[Position, list[dict]] = {p: [] for p in Position}

        for player_element in all_elements:
            player_id = player_element["id"]
            position = self.api_client.get_player_position(player_element)
            data = player_info_map[player_id]
            attributes = PlayerAttributes(**player_element)

            raw_player = RawPlayer(
                player_id=player_id,
                position=position,
                attributes=attributes,
                **data,
            )

            # Current inference features
            features = PlayerFeatureTransformer.transform(raw_player)
            features_by_position[position].append(features.model_dump())

            # Historical training rows
            examples = PlayerFeatureTransformer.transform_history(raw_player)

            for example in examples:
                training_example = TrainingExample(**example)
                training_by_position[position].append(json.loads(training_example.model_dump_json()))

        print(f"Loaded {num_players} players from FPL API")

        # Upsert to MongoDB
        for position in Position:
            feat_count = self.store.upsert_player_features(position, features_by_position[position])
            if feat_count:
                print(f"  {POSITION_NAMES[position]}: {feat_count} features stored")

            train_count = self.store.upsert_training_examples(position, season, training_by_position[position])
            if train_count:
                print(f"  {POSITION_NAMES[position]}: {train_count} training examples stored for {season}")

        print(f"\nTraining data stored in MongoDB for season: {season}")

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

        gp = GPModel(store=self.store)
        gp.train()
        gp.predict()

    def optimise_team(self, user_id: int, max_transfers: int = 2):
        """Optimise a user's team by suggesting transfers that maximise xP.

        Returns the OptimisationResult (also prints a summary to stdout).
        """
        from src.optimisation.team_optimiser import TeamOptimiser

        optimiser = TeamOptimiser(store=self.store)
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

        return result
