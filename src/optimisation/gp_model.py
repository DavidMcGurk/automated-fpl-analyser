"""Gaussian Process model for predicting FPL player expected points (xP).

Trains per-position GP regression models using gpytorch on historical
training data, then predicts xP for current players from inference features.
"""

import json
from pathlib import Path

import torch
import gpytorch

from src.models.post_prediction import Position
from src.models.player_features import (
    AttackerFeatures,
    DefenderFeatures,
    GoalkeeperFeatures,
    MidfielderFeatures,
)

BASE_DIR = Path(__file__).resolve().parents[2]
TRAINING_BASE_DIR = BASE_DIR / "data/training"
FEATURE_DIR = BASE_DIR / "data/player_features"
PREDICTION_DIR = BASE_DIR / "data/predictions"

POSITION_NAMES = {
    Position.GOALKEEPER: "goalkeepers",
    Position.DEFENDER: "defenders",
    Position.MIDFIELDER: "midfielders",
    Position.ATTACKER: "attackers",
}

FEATURE_MODELS = {
    Position.GOALKEEPER: GoalkeeperFeatures,
    Position.DEFENDER: DefenderFeatures,
    Position.MIDFIELDER: MidfielderFeatures,
    Position.ATTACKER: AttackerFeatures,
}

EXCLUDE_FEATURES = {"player_id"}

TRAINING_EPOCHS = 100
LEARNING_RATE = 0.05
MAX_TRAIN_SAMPLES = 2000


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1]))
        # Initialize lengthscale to a small value so the kernel is sensitive to feature differences
        self.covar_module.base_kernel.lengthscale = 1.0

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class GPModel:
    """Manages training and prediction of per-position GP models."""

    def __init__(self) -> None:
        self.models: dict[Position, GPRegressionModel] = {}
        self.likelihoods: dict[Position, gpytorch.likelihoods.GaussianLikelihood] = {}
        self.feature_columns: dict[Position, list[str]] = {}
        self.feature_means: dict[Position, torch.Tensor] = {}
        self.feature_stds: dict[Position, torch.Tensor] = {}

    def train(self, seasons: list[str] | None = None) -> None:
        """Train GP models for all positions using historical training data.

        Args:
            seasons: List of season folder names to train on (e.g. ["2024_25"]).
                     If None, uses all available seasons.
        """
        if seasons is None:
            seasons = sorted(d.name for d in TRAINING_BASE_DIR.iterdir() if d.is_dir())

        for position in Position:
            print(f"\n=== Training {POSITION_NAMES[position]} GP ===")

            train_x, train_y, columns = self._load_training_data(position, seasons)

            if len(train_x) == 0:
                print(f"  No training data for {POSITION_NAMES[position]}, skipping")
                continue

            print(f"  Loaded {len(train_x)} training examples with {len(columns)} features")

            self._train_position(position, train_x, train_y)
            self.feature_columns[position] = columns

    def predict(self) -> None:
        """Predict xP for all current players and write to data/predictions/."""
        PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

        for position in Position:
            if position not in self.models:
                print(f"No model for {POSITION_NAMES[position]}, skipping prediction")
                continue

            feature_path = FEATURE_DIR / f"{POSITION_NAMES[position]}.jsonl"
            if not feature_path.exists():
                print(f"No features for {POSITION_NAMES[position]}, skipping")
                continue

            predictions = self._predict_position(position, feature_path)

            output_path = PREDICTION_DIR / f"{POSITION_NAMES[position]}.jsonl"
            with output_path.open("w") as f:
                for pred in predictions:
                    f.write(json.dumps(pred) + "\n")

            print(f"  {POSITION_NAMES[position]}: {len(predictions)} predictions -> {output_path}")

    def _load_training_data(
        self, position: Position, seasons: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """Load and prepare training data for a given position."""
        examples: list[dict] = []

        for season in seasons:
            path = TRAINING_BASE_DIR / season / f"{POSITION_NAMES[position]}.jsonl"
            if not path.exists():
                continue

            with path.open() as f:
                for line in f:
                    if line.strip():
                        examples.append(json.loads(line))

        if not examples:
            return torch.empty(0), torch.empty(0), []

        feature_model = FEATURE_MODELS[position]
        all_columns = list(feature_model.model_fields.keys())
        columns = [c for c in all_columns if c not in EXCLUDE_FEATURES]

        x_list = []
        y_list = []

        for ex in examples:
            features = ex.get("features", {})
            row = []
            for col in columns:
                val = features.get(col)
                row.append(0.0 if val is None else float(val))
            x_list.append(row)
            y_list.append(float(ex.get("target_points", 0)))

        x_tensor = torch.tensor(x_list, dtype=torch.float32)
        y_tensor = torch.tensor(y_list, dtype=torch.float32)

        return x_tensor, y_tensor, columns

    def _train_position(self, position: Position, train_x: torch.Tensor, train_y: torch.Tensor) -> None:
        """Train a GP model for a single position."""
        # Standardize features (zero mean, unit variance)
        mean = train_x.mean(dim=0)
        std = train_x.std(dim=0)
        std[std == 0] = 1.0  # Avoid division by zero for constant features
        train_x = (train_x - mean) / std

        self.feature_means[position] = mean
        self.feature_stds[position] = std

        if len(train_x) > MAX_TRAIN_SAMPLES:
            print(f"  Subsampling from {len(train_x)} to {MAX_TRAIN_SAMPLES} for tractability")
            indices = torch.randperm(len(train_x))[:MAX_TRAIN_SAMPLES]
            train_x = train_x[indices]
            train_y = train_y[indices]

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPRegressionModel(train_x, train_y, likelihood)

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for epoch in range(TRAINING_EPOCHS):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{TRAINING_EPOCHS} - Loss: {loss.item():.3f}")

        model.eval()
        likelihood.eval()

        self.models[position] = model
        self.likelihoods[position] = likelihood

    def _predict_position(self, position: Position, feature_path: Path) -> list[dict]:
        """Predict xP for all players at a given position."""
        model = self.models[position]
        likelihood = self.likelihoods[position]
        columns = self.feature_columns[position]
        mean = self.feature_means[position]
        std = self.feature_stds[position]

        predictions = []

        with feature_path.open() as f:
            for line in f:
                if not line.strip():
                    continue

                features_dict = json.loads(line)
                player_id = features_dict.get("player_id")

                row = []
                for col in columns:
                    val = features_dict.get(col)
                    row.append(0.0 if val is None else float(val))

                x = torch.tensor([row], dtype=torch.float32)
                x = (x - mean) / std  # Apply same standardization

                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    pred_dist = likelihood(model(x))
                    mean = pred_dist.mean.item()
                    variance = pred_dist.variance.item()

                predictions.append(
                    {
                        "player_id": player_id,
                        "position": position.value,
                        "xp": round(mean, 2),
                        "xp_uncertainty": round(variance**0.5, 2),
                    }
                )

        return predictions
