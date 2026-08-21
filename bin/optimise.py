"""CLI tool for FPL team optimisation.

Usage:
    uv run python bin/optimise.py <user_id> [--max-transfers N]

Arguments:
    user_id          Your FPL user ID (find it at fantasy.premierleague.com,
                     in the URL of your team page)

Options:
    --max-transfers   Maximum transfers to consider (default: 2)
    --train           Retrain the GP model before optimising
"""

import argparse

from src.optimisation.predictor import Predictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimise your FPL team using Gaussian Process xP predictions.")
    parser.add_argument(
        "user_id",
        type=int,
        help="Your FPL user ID (from your team page URL)",
    )
    parser.add_argument(
        "--max-transfers",
        type=int,
        default=2,
        help="Maximum transfers to consider (default: 2)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Retrain the GP model before optimising",
    )

    args = parser.parse_args()

    predictor = Predictor()

    if args.train:
        print("Loading player data...")
        predictor.load_player_data()

        print("\nTraining GP model...")
        predictor.model_xp()

    print(f"\nOptimising team for user {args.user_id}...")
    predictor.optimise_team(args.user_id, max_transfers=args.max_transfers)


if __name__ == "__main__":
    main()
