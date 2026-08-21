"""Tests for PlayerFeatureTransformer."""

from src.optimisation.player_feature_transformer import PlayerFeatureTransformer
from src.models.player_features import (
    GoalkeeperFeatures,
    DefenderFeatures,
    MidfielderFeatures,
    AttackerFeatures,
)
from src.models.post_prediction import Position


class TestTransform:
    def test_goalkeeper_transform(self, make_raw_player):
        player = make_raw_player(player_id=1, position_type=1)
        features = PlayerFeatureTransformer.transform(player)
        assert isinstance(features, GoalkeeperFeatures)
        assert features.player_id == 1
        assert features.playing_probability == 1.0

    def test_defender_transform(self, make_raw_player):
        player = make_raw_player(player_id=2, position_type=2)
        features = PlayerFeatureTransformer.transform(player)
        assert isinstance(features, DefenderFeatures)

    def test_midfielder_transform(self, make_raw_player):
        player = make_raw_player(player_id=3, position_type=3)
        features = PlayerFeatureTransformer.transform(player)
        assert isinstance(features, MidfielderFeatures)

    def test_attacker_transform(self, make_raw_player):
        player = make_raw_player(player_id=4, position_type=4)
        features = PlayerFeatureTransformer.transform(player)
        assert isinstance(features, AttackerFeatures)

    def test_empty_history(self, make_raw_player):
        """Transform with no history should still produce valid features."""
        player = make_raw_player(history=[])
        features = PlayerFeatureTransformer.transform(player)
        assert features.avg_points_last_3 is None
        assert features.avg_minutes_last_3 is None


class TestTransformHistory:
    def test_no_history_returns_empty(self, make_raw_player):
        """With no history, transform_history should return no examples."""
        player = make_raw_player(history=[])
        examples = PlayerFeatureTransformer.transform_history(player)
        assert examples == []

    def test_single_gameweek_returns_empty(self, make_raw_player, make_history):
        """With one gameweek, there's no 'next' gameweek to predict."""
        player = make_raw_player(history=[make_history()])
        examples = PlayerFeatureTransformer.transform_history(player)
        assert examples == []

    def test_two_gameweeks_returns_one_example(self, make_raw_player, make_history):
        """Two gameweeks should produce one training example."""
        player = make_raw_player(history=[make_history(total_points=5), make_history(total_points=8)])
        examples = PlayerFeatureTransformer.transform_history(player)
        assert len(examples) == 1
        assert examples[0]["gameweek"] == 1
        assert examples[0]["target_points"] == 8
        assert examples[0]["player_id"] == player.player_id

    def test_three_gameweeks_returns_two_examples(self, make_raw_player, make_history):
        player = make_raw_player(
            history=[
                make_history(total_points=3),
                make_history(total_points=5),
                make_history(total_points=7),
            ]
        )
        examples = PlayerFeatureTransformer.transform_history(player)
        assert len(examples) == 2
        assert examples[0]["gameweek"] == 1
        assert examples[0]["target_points"] == 5
        assert examples[1]["gameweek"] == 2
        assert examples[1]["target_points"] == 7

    def test_position_in_examples(self, make_raw_player, make_history):
        player = make_raw_player(player_id=5, position_type=4)
        examples = PlayerFeatureTransformer.transform_history(
            player._replace() if hasattr(player, "_replace") else player
        )
        # Need at least 2 history entries
        from src.models.pre_processing import RawPlayer

        player = RawPlayer(
            player_id=5,
            position=Position.ATTACKER,
            attributes=player.attributes,
            fixtures=[],
            history=[make_history(), make_history()],
            history_past=[],
        )
        examples = PlayerFeatureTransformer.transform_history(player)
        assert len(examples) == 1
        assert examples[0]["position"] == Position.ATTACKER
