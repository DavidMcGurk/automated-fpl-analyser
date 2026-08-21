"""CLI tool for GP model validation and hyperparameter experiments.

Usage:
    # Validate with default settings (exact GP, RBF kernel, normalized targets)
    uv run python bin/validate.py

    # Compare kernels
    uv run python bin/validate.py --kernel matern52

    # Try SVGP with Matérn 3/2
    uv run python bin/validate.py --model svgp --kernel matern32

    # Train on specific seasons, validate on another
    uv run python bin/validate.py --train-seasons 2022_23 2023_24 --val-seasons 2024_25
"""

import argparse
import json

from src.optimisation.gp_model import GPModel, POSITION_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate GP model hyperparameters.")
    parser.add_argument(
        "--model",
        choices=["exact", "svgp"],
        default="exact",
        help="Model type: exact GP (subsampling) or SVGP (inducing points)",
    )
    parser.add_argument(
        "--kernel",
        choices=["rbf", "matern32", "matern52"],
        default="rbf",
        help="Kernel function",
    )
    parser.add_argument(
        "--no-normalize-target",
        action="store_true",
        help="Disable target normalization",
    )
    parser.add_argument(
        "--train-seasons",
        nargs="+",
        default=None,
        help="Seasons to train on (default: all except last)",
    )
    parser.add_argument(
        "--val-seasons",
        nargs="+",
        default=None,
        help="Seasons to validate on (default: last season)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write results to JSON file (default: print only)",
    )

    args = parser.parse_args()

    # Default: train on all seasons except the last, validate on the last
    if args.train_seasons is None or args.val_seasons is None:
        from pathlib import Path

        BASE_DIR = Path(__file__).resolve().parents[1]
        training_dir = BASE_DIR / "data/training"
        all_seasons = sorted(d.name for d in training_dir.iterdir() if d.is_dir())

        # Filter out empty seasons (current season with no games yet)
        non_empty = []
        for s in all_seasons:
            has_data = any(
                (training_dir / s / f"{pos}.jsonl").stat().st_size > 0
                for pos in ["goalkeepers", "defenders", "midfielders", "attackers"]
                if (training_dir / s / f"{pos}.jsonl").exists()
            )
            if has_data:
                non_empty.append(s)

        if len(non_empty) < 2:
            print("Need at least 2 seasons with data for validation.")
            return

        if args.train_seasons is None:
            args.train_seasons = non_empty[:-1]
        if args.val_seasons is None:
            args.val_seasons = [non_empty[-1]]

    print(f"Model: {args.model}, Kernel: {args.kernel}, Normalize target: {not args.no_normalize_target}")
    print(f"Train seasons: {args.train_seasons}")
    print(f"Validation seasons: {args.val_seasons}")

    model = GPModel(
        model_type=args.model,
        kernel_name=args.kernel,
        normalize_target=not args.no_normalize_target,
    )

    results = model.validate(args.train_seasons, args.val_seasons)

    # Summary
    print(f"\n{'=' * 60}")
    print("Validation Summary")
    print(f"{'=' * 60}")
    print(f"{'Position':<15} {'RMSE':>8} {'MAE':>8} {'Mean Pred':>10} {'Mean Actual':>12}")
    print("-" * 55)
    for position, metrics in results.items():
        print(
            f"{POSITION_NAMES[position]:<15} {metrics['rmse']:>8.3f} {metrics['mae']:>8.3f} "
            f"{metrics['mean_pred']:>10.3f} {metrics['mean_actual']:>12.3f}"
        )

    # Overall averages
    if results:
        avg_rmse = sum(m["rmse"] for m in results.values()) / len(results)
        avg_mae = sum(m["mae"] for m in results.values()) / len(results)
        print("-" * 55)
        print(f"{'Average':<15} {avg_rmse:>8.3f} {avg_mae:>8.3f}")

    if args.output:
        output_data = {
            "model": args.model,
            "kernel": args.kernel,
            "normalize_target": not args.no_normalize_target,
            "train_seasons": args.train_seasons,
            "val_seasons": args.val_seasons,
            "results": {POSITION_NAMES[pos]: metrics for pos, metrics in results.items()},
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
