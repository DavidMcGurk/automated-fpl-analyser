"""Tests for MongoDB storage layer."""

from unittest.mock import MagicMock, patch

import pytest

from src.models.post_prediction import Position
from src.storage.mongo_client import MongoStore


class TestMongoStoreInit:
    def test_requires_uri(self):
        """MongoStore should raise if no URI is provided."""
        import os

        old_val = os.environ.pop("MONGODB_URI", None)
        try:
            with pytest.raises(ValueError, match="MONGODB_URI"):
                MongoStore()
        finally:
            if old_val:
                os.environ["MONGODB_URI"] = old_val

    @patch("src.storage.mongo_client.MongoClient")
    def test_init_with_uri(self, mock_client_class):
        store = MongoStore(uri="mongodb://localhost:27017")
        mock_client_class.assert_called_once_with("mongodb://localhost:27017")
        assert store.db is not None

    @patch("src.storage.mongo_client.MongoClient")
    def test_init_from_env(self, mock_client_class):
        import os

        os.environ["MONGODB_URI"] = "mongodb://localhost:27017"
        try:
            MongoStore()
            mock_client_class.assert_called_once_with("mongodb://localhost:27017")
        finally:
            del os.environ["MONGODB_URI"]


class TestUpsertTrainingExamples:
    @patch("src.storage.mongo_client.MongoClient")
    def test_upsert_empty_list(self, mock_client_class):
        store = MongoStore(uri="mongodb://localhost:27017")
        result = store.upsert_training_examples(Position.GOALKEEPER, "2024_25", [])
        assert result == 0

    @patch("src.storage.mongo_client.MongoClient")
    def test_upsert_calls_bulk_write(self, mock_client_class):
        mock_db = MagicMock()
        mock_client_class.return_value.__getitem__.return_value = mock_db
        mock_collection = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        mock_result = MagicMock()
        mock_result.upserted_count = 2
        mock_result.modified_count = 0
        mock_collection.bulk_write.return_value = mock_result

        store = MongoStore(uri="mongodb://localhost:27017")
        examples = [
            {"player_id": 1, "gameweek": 1, "features": {}, "target_points": 5},
            {"player_id": 1, "gameweek": 2, "features": {}, "target_points": 3},
        ]
        result = store.upsert_training_examples(Position.GOALKEEPER, "2024_25", examples)
        assert result == 2
        mock_collection.bulk_write.assert_called_once()


class TestLoadTrainingExamples:
    @patch("src.storage.mongo_client.MongoClient")
    def test_load_returns_list(self, mock_client_class):
        mock_db = MagicMock()
        mock_client_class.return_value.__getitem__.return_value = mock_db
        mock_collection = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        mock_collection.find.return_value = [
            {"player_id": 1, "target_points": 5},
            {"player_id": 2, "target_points": 3},
        ]

        store = MongoStore(uri="mongodb://localhost:27017")
        result = store.load_training_examples(Position.GOALKEEPER, ["2024_25"])
        assert len(result) == 2
        assert result[0]["player_id"] == 1


class TestListSeasons:
    @patch("src.storage.mongo_client.MongoClient")
    def test_list_seasons_sorted(self, mock_client_class):
        mock_db = MagicMock()
        mock_client_class.return_value.__getitem__.return_value = mock_db
        mock_collection = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        mock_collection.distinct.return_value = ["2024_25", "2022_23", "2023_24"]

        store = MongoStore(uri="mongodb://localhost:27017")
        result = store.list_seasons()
        assert result == ["2022_23", "2023_24", "2024_25"]


class TestLoadAllPredictions:
    @patch("src.storage.mongo_client.MongoClient")
    def test_load_all_predictions_keyed_by_player_id(self, mock_client_class):
        mock_db = MagicMock()
        mock_client_class.return_value.__getitem__.return_value = mock_db
        mock_collection = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        mock_collection.find.return_value = [
            {"player_id": 1, "xp": 5.0},
            {"player_id": 2, "xp": 3.0},
        ]

        store = MongoStore(uri="mongodb://localhost:27017")
        result = store.load_all_predictions()
        assert result[1]["xp"] == 5.0
        assert result[2]["xp"] == 3.0


class TestUpsertPredictions:
    @patch("src.storage.mongo_client.MongoClient")
    def test_upsert_empty_list(self, mock_client_class):
        mock_db = MagicMock()
        mock_client_class.return_value.__getitem__.return_value = mock_db
        mock_collection = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        mock_collection.count_documents.return_value = 0

        store = MongoStore(uri="mongodb://localhost:27017")
        result = store.upsert_predictions(Position.DEFENDER, [])
        assert result == 0
        mock_collection.delete_many.assert_called_once()

    @patch("src.storage.mongo_client.MongoClient")
    def test_upsert_calls_bulk_write(self, mock_client_class):
        mock_db = MagicMock()
        mock_client_class.return_value.__getitem__.return_value = mock_db
        mock_collection = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        mock_collection.count_documents.return_value = 1

        store = MongoStore(uri="mongodb://localhost:27017")
        predictions = [{"player_id": 1, "xp": 5.0}]
        result = store.upsert_predictions(Position.DEFENDER, predictions)
        assert result == 1
        mock_collection.bulk_write.assert_called_once()
        mock_collection.delete_many.assert_called_once()


class TestUpsertPlayerFeatures:
    @patch("src.storage.mongo_client.MongoClient")
    def test_upsert_empty_list(self, mock_client_class):
        mock_db = MagicMock()
        mock_client_class.return_value.__getitem__.return_value = mock_db
        mock_collection = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        mock_collection.count_documents.return_value = 0

        store = MongoStore(uri="mongodb://localhost:27017")
        result = store.upsert_player_features(Position.MIDFIELDER, [])
        assert result == 0
        mock_collection.delete_many.assert_called_once()

    @patch("src.storage.mongo_client.MongoClient")
    def test_upsert_deletes_stale_features(self, mock_client_class):
        """Stale features for players no longer in the API should be deleted."""
        mock_db = MagicMock()
        mock_client_class.return_value.__getitem__.return_value = mock_db
        mock_collection = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        mock_collection.count_documents.return_value = 2

        store = MongoStore(uri="mongodb://localhost:27017")
        features = [
            {"player_id": 1, "now_cost": 5.0},
            {"player_id": 2, "now_cost": 6.0},
        ]
        result = store.upsert_player_features(Position.MIDFIELDER, features)
        assert result == 2

        # Verify delete_many was called with $nin for stale player IDs
        delete_call = mock_collection.delete_many.call_args
        delete_filter = delete_call[0][0]
        assert delete_filter["position"] == Position.MIDFIELDER.value
        assert "$nin" in delete_filter["player_id"]
        assert set(delete_filter["player_id"]["$nin"]) == {1, 2}
