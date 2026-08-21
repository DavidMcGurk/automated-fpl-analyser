"""MongoDB storage layer for FPL analyser data.

Collections:
- training_examples: per-gameweek training rows (season + position indexed)
- player_features: current inference features per player (position indexed)
- predictions: xP predictions per player (position indexed)

All collections use a compound key for upserts to avoid duplicates.
"""

import json
import os
from typing import Any

from pymongo import ASCENDING, MongoClient, UpdateOne

from src.models.post_prediction import Position

POSITION_NAMES = {
    Position.GOALKEEPER: "goalkeepers",
    Position.DEFENDER: "defenders",
    Position.MIDFIELDER: "midfielders",
    Position.ATTACKER: "attackers",
}


class MongoStore:
    """Wraps MongoDB operations for training data, features, and predictions."""

    def __init__(self, uri: str | None = None, db_name: str = "fpl_analyser") -> None:
        uri = uri or os.environ.get("MONGODB_URI")
        if not uri:
            raise ValueError("MONGODB_URI not set. Set it as an environment variable or pass uri= to MongoStore.")
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

        self.training = self.db["training_examples"]
        self.features = self.db["player_features"]
        self.predictions = self.db["predictions"]

        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.training.create_index(
            [("season", ASCENDING), ("position", ASCENDING), ("player_id", ASCENDING), ("gameweek", ASCENDING)],
            unique=True,
        )
        self.features.create_index(
            [("position", ASCENDING), ("player_id", ASCENDING)],
            unique=True,
        )
        self.predictions.create_index(
            [("position", ASCENDING), ("player_id", ASCENDING)],
            unique=True,
        )

    def close(self) -> None:
        self.client.close()

    # -- Training examples --

    def upsert_training_examples(self, position: Position, season: str, examples: list[dict]) -> int:
        """Bulk upsert training examples for a position + season."""
        if not examples:
            return 0

        ops = []
        for ex in examples:
            filter_doc = {
                "season": season,
                "position": position.value,
                "player_id": ex.get("player_id"),
                "gameweek": ex.get("gameweek"),
            }
            doc = {**filter_doc, **ex}
            ops.append(UpdateOne(filter_doc, {"$set": doc}, upsert=True))

        result = self.training.bulk_write(ops)
        return result.upserted_count + result.modified_count

    def load_training_examples(self, position: Position, seasons: list[str]) -> list[dict]:
        """Load training examples for a position across given seasons."""
        cursor = self.training.find(
            {"position": position.value, "season": {"$in": seasons}},
            {"_id": 0},
        )
        return list(cursor)

    def list_seasons(self) -> list[str]:
        """Return sorted list of seasons that have training data."""
        seasons = self.training.distinct("season")
        return sorted(seasons)

    def count_training_examples(self, position: Position, season: str) -> int:
        return self.training.count_documents({"position": position.value, "season": season})

    # -- Player features --

    def upsert_player_features(self, position: Position, features: list[dict]) -> int:
        """Bulk upsert current player features for a position."""
        if not features:
            return 0

        ops = []
        for feat in features:
            player_id = feat.get("player_id")
            filter_doc = {"position": position.value, "player_id": player_id}
            doc = {**filter_doc, **feat}
            ops.append(UpdateOne(filter_doc, {"$set": doc}, upsert=True))

        result = self.features.bulk_write(ops)
        return result.upserted_count + result.modified_count

    def load_player_features(self, position: Position) -> list[dict]:
        """Load all current player features for a position."""
        cursor = self.features.find({"position": position.value}, {"_id": 0})
        return list(cursor)

    # -- Predictions --

    def upsert_predictions(self, position: Position, predictions: list[dict]) -> int:
        """Bulk upsert xP predictions for a position."""
        if not predictions:
            return 0

        ops = []
        for pred in predictions:
            player_id = pred.get("player_id")
            filter_doc = {"position": position.value, "player_id": player_id}
            doc = {**filter_doc, **pred}
            ops.append(UpdateOne(filter_doc, {"$set": doc}, upsert=True))

        result = self.predictions.bulk_write(ops)
        return result.upserted_count + result.modified_count

    def load_predictions(self, position: Position) -> list[dict]:
        """Load all predictions for a position."""
        cursor = self.predictions.find({"position": position.value}, {"_id": 0})
        return list(cursor)

    def load_all_predictions(self) -> dict[int, dict]:
        """Load all predictions across positions, keyed by player_id."""
        cursor = self.predictions.find({}, {"_id": 0})
        return {pred["player_id"]: pred for pred in cursor}

    # -- Migration helper --

    def migrate_from_jsonl(
        self, jsonl_path: str, collection: str, position: Position, season: str | None = None
    ) -> int:
        """Import records from a JSONL file into a MongoDB collection."""
        with open(jsonl_path) as f:
            records: list[dict[str, Any]] = []
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        if collection == "training_examples":
            if season is None:
                raise ValueError("season is required for training_examples migration")
            return self.upsert_training_examples(position, season, records)
        elif collection == "player_features":
            return self.upsert_player_features(position, records)
        elif collection == "predictions":
            return self.upsert_predictions(position, records)
        else:
            raise ValueError(f"Unknown collection: {collection}")
