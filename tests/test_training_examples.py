"""Tests for training examples model."""

from src.models.training_examples import TrainingExample
from src.models.post_prediction import Position
from src.models.player_features import GoalkeeperFeatures, AttackerFeatures


class TestTrainingExample:
    def test_goalkeeper_example(self):
        features = GoalkeeperFeatures(
            player_id=1,
            playing_probability=1.0,
            next_fixture_difficulty=2.0,
            avg_fixture_difficulty_3=2.0,
            avg_fixture_difficulty_5=None,
            home_fixture_ratio_next_5=0.5,
            avg_points_last_3=3.0,
            avg_points_last_5=None,
            avg_minutes_last_3=90.0,
            avg_minutes_last_5=None,
            yellow_cards_last_5=0.0,
            red_cards_last_5=0.0,
            selected_by_percent=10.0,
            transfers_balance_last_5=None,
            now_cost=5.0,
            avg_price_diff_historic=None,
            avg_points_per_90_historic=None,
            avg_minutes_per_season_historic=None,
            saves_per_90=2.0,
            clean_sheets_per_90=0.3,
            saves_per_90_last_3=None,
            saves_per_90_last_5=None,
            goals_conceded_per_90=1.0,
            penalties_saved=0,
        )
        example = TrainingExample(
            player_id=1,
            position=Position.GOALKEEPER,
            gameweek=5,
            features=features,
            target_points=4,
        )
        assert example.player_id == 1
        assert example.gameweek == 5
        assert example.target_points == 4
        assert isinstance(example.features, GoalkeeperFeatures)

    def test_attacker_example(self):
        features = AttackerFeatures(
            player_id=10,
            playing_probability=0.8,
            next_fixture_difficulty=3.0,
            avg_fixture_difficulty_3=None,
            avg_fixture_difficulty_5=None,
            home_fixture_ratio_next_5=None,
            avg_points_last_3=None,
            avg_points_last_5=None,
            avg_minutes_last_3=None,
            avg_minutes_last_5=None,
            yellow_cards_last_5=None,
            red_cards_last_5=None,
            selected_by_percent=50.0,
            transfers_balance_last_5=None,
            now_cost=10.0,
            avg_price_diff_historic=None,
            avg_points_per_90_historic=None,
            avg_minutes_per_season_historic=None,
            expected_goal_involvements_per_90=0.5,
            avg_xg_last_3=None,
            avg_xg_last_5=None,
            avg_xa_last_3=None,
            avg_xa_last_5=None,
            goals_per_90=None,
            assists_per_90=None,
            avg_set_piece_order=None,
        )
        example = TrainingExample(
            player_id=10,
            position=Position.ATTACKER,
            gameweek=1,
            features=features,
            target_points=8,
        )
        assert example.position == Position.ATTACKER
        assert isinstance(example.features, AttackerFeatures)
