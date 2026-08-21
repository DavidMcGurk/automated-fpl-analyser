"""One-time migration tool: imports existing JSONL files into MongoDB.

Usage:
    # Migrate all training data, features, and predictions
    uv run python bin/migrate.py

    # Migrate only training data for specific seasons
    uv run python bin/migrate.py --only training --seasons 2023_24 2024_25

    # Migrate only current features and predictions
    uv run python bin/migrate.py --only features predictions
"""

import argparse
from pathlib import Path

from src.models.post_prediction import Position
from src.storage.mongo_client import MongoStore, POSITION_NAMES

BASE_DIR = Path(__file__).resolve().parents[1]
TRAINING_BASE_DIR = BASE_DIR / "data/training"
FEATURE_DIR = BASE_DIR / "data/player_features"
PREDICTION_DIR = BASE_DIR / "data/predictions"


def migrate_training(store: MongoStore, seasons: list[str] | None = None) -> None:
    """Migrate training JSONL files to MongoDB."""
    if seasons is None:
        seasons = sorted(d.name for d in TRAINING_BASE_DIR.iterdir() if d.is_dir())

    for season in seasons:
        season_dir = TRAINING_BASE_DIR / season
        if not season_dir.exists():
            print(f"  Season {season}: directory not found, skipping")
            continue

        for position in Position:
            path = season_dir / f"{POSITION_NAMES[position]}.jsonl"
            if not path.exists() or path.stat().st_size == 0:
                continue

            count = store.migrate_from_jsonl(str(path), "training_examples", position, season)
            print(f"  {season}/{POSITION_NAMES[position]}: {count} training examples migrated")


def migrate_features(store: MongoStore) -> None:
    """Migrate player features JSONL to MongoDB."""
    for position in Position:
        path = FEATURE_DIR / f"{POSITION_NAMES[position]}.jsonl"
        if not path.exists() or path.stat().st_size == 0:
            continue

        count = store.migrate_from_jsonl(str(path), "player_features", position)
        print(f"  {POSITION_NAMES[position]}: {count} features migrated")


def migrate_predictions(store: MongoStore) -> None:
    """Migrate predictions JSONL to MongoDB."""
    for position in Position:
        path = PREDICTION_DIR / f"{POSITION_NAMES[position]}.jsonl"
        if not path.exists() or path.stat().st_size == 0:
            continue

        count = store.migrate_from_jsonl(str(path), "predictions", position)
        print(f"  {POSITION_NAMES[position]}: {count} predictions migrated")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate JSONL files to MongoDB.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["training", "features", "predictions"],
        default=["training", "features", "predictions"],
        help="What to migrate (default: all)",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=None,
        help="Specific seasons to migrate training data for (default: all)",
    )

    args = parser.parse_args()

    store = MongoStore()

    try:
        if "training" in args.only:
            print("\n=== Migrating training data ===")
            migrate_training(store, args.seasons)

        if "features" in args.only:
            print("\n=== Migrating player features ===")
            migrate_features(store)

        if "predictions" in args.only:
            print("\n=== Migrating predictions ===")
            migrate_predictions(store)

        print("\nMigration complete!")
    finally:
        store.close()


if __name__ == "__main__":
    main()
