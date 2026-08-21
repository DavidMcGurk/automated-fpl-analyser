"""Gaussian Process model for predicting FPL player expected points (xP).

Trains per-position GP regression models using gpytorch on historical
training data, then predicts xP for current players from inference features.

Supports two model types:
- "exact": Exact GP with subsampling (default, O(n³) but simple)
- "svgp": Stochastic Variational GP with inducing points (scales to large datasets)

Kernel can be configured: "rbf", "matern32", or "matern52".
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
NUM_INDUCING_POINTS = 500
BATCH_SIZE = 256

KERNELS = {
    "rbf": gpytorch.kernels.RBFKernel,
    "matern32": gpytorch.kernels.MaternKernel,
    "matern52": gpytorch.kernels.MaternKernel,
}


def _make_base_kernel(kernel_name: str, num_dims: int) -> gpytorch.kernels.Kernel:
    """Create a base kernel by name with ARD."""
    if kernel_name == "rbf":
        return KERNELS["rbf"](ard_num_dims=num_dims)
    elif kernel_name == "matern32":
        return KERNELS["matern32"](nu=1.5, ard_num_dims=num_dims)
    elif kernel_name == "matern52":
        return KERNELS["matern52"](nu=2.5, ard_num_dims=num_dims)
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}. Use 'rbf', 'matern32', or 'matern52'.")


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_name="rbf"):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_kernel = _make_base_kernel(kernel_name, train_x.shape[-1])
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
        self.covar_module.base_kernel.lengthscale = 1.0

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class SVGPModel(gpytorch.models.ApproximateGP):
    """Stochastic Variational GP for scalable training on large datasets."""

    def __init__(self, inducing_points, kernel_name="rbf"):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=len(inducing_points)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        base_kernel = _make_base_kernel(kernel_name, inducing_points.shape[-1])
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class GPModel:
    """Manages training and prediction of per-position GP models.

    Args:
        model_type: "exact" for ExactGP (subsampling) or "svgp" for variational GP.
        kernel_name: "rbf", "matern32", or "matern52".
        normalize_target: If True, standardize target points to zero mean / unit variance.
    """

    def __init__(
        self,
        model_type: str = "exact",
        kernel_name: str = "rbf",
        normalize_target: bool = True,
    ) -> None:
        self.model_type = model_type
        self.kernel_name = kernel_name
        self.normalize_target = normalize_target

        self.models: dict[Position, torch.nn.Module] = {}
        self.likelihoods: dict[Position, torch.nn.Module] = {}
        self.feature_columns: dict[Position, list[str]] = {}
        self.feature_means: dict[Position, torch.Tensor] = {}
        self.feature_stds: dict[Position, torch.Tensor] = {}
        self.target_means: dict[Position, float] = {}
        self.target_stds: dict[Position, float] = {}

    def train(self, seasons: list[str] | None = None) -> None:
        """Train GP models for all positions using historical training data.

        Args:
            seasons: List of season folder names to train on (e.g. ["2024_25"]).
                     If None, uses all available seasons.
        """
        if seasons is None:
            seasons = sorted(d.name for d in TRAINING_BASE_DIR.iterdir() if d.is_dir())

        for position in Position:
            print(f"\n=== Training {POSITION_NAMES[position]} GP ({self.model_type}, {self.kernel_name}) ===")

            train_x, train_y, columns = self._load_training_data(position, seasons)

            if len(train_x) == 0:
                print(f"  No training data for {POSITION_NAMES[position]}, skipping")
                continue

            print(f"  Loaded {len(train_x)} training examples with {len(columns)} features")

            self._train_position(position, train_x, train_y)
            self.feature_columns[position] = columns

    def validate(
        self,
        train_seasons: list[str],
        val_seasons: list[str],
    ) -> dict[Position, dict[str, float]]:
        """Walk-forward validation: train on train_seasons, evaluate on val_seasons.

        Returns per-position metrics: RMSE, MAE, and mean prediction.
        """
        results: dict[Position, dict[str, float]] = {}

        for position in Position:
            print(f"\n=== Validating {POSITION_NAMES[position]} ===")

            val_x, val_y, _ = self._load_training_data(position, val_seasons)
            if len(val_x) == 0:
                print(f"  No validation data for {POSITION_NAMES[position]}, skipping")
                continue

            train_x, train_y, columns = self._load_training_data(position, train_seasons)
            if len(train_x) == 0:
                print(f"  No training data for {POSITION_NAMES[position]}, skipping")
                continue

            self._train_position(position, train_x, train_y)
            self.feature_columns[position] = columns

            # Predict on validation set
            mean = self.feature_means[position]
            std = self.feature_stds[position]
            val_x_norm = (val_x - mean) / std

            model = self.models[position]
            likelihood = self.likelihoods[position]
            model.eval()
            likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred_dist = likelihood(model(val_x_norm))
                preds = pred_dist.mean

            # Denormalize predictions if target was normalized
            if self.normalize_target:
                t_mean = self.target_means[position]
                t_std = self.target_stds[position]
                preds = preds * t_std + t_mean

            preds = preds.numpy()
            actuals = val_y.numpy()

            rmse = float((((preds - actuals) ** 2).mean()) ** 0.5)
            mae = float((abs(preds - actuals)).mean())
            mean_pred = float(preds.mean())
            mean_actual = float(actuals.mean())

            print(f"  RMSE: {rmse:.3f}, MAE: {mae:.3f}, Mean pred: {mean_pred:.3f}, Mean actual: {mean_actual:.3f}")

            results[position] = {
                "rmse": rmse,
                "mae": mae,
                "mean_pred": mean_pred,
                "mean_actual": mean_actual,
            }

        return results

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
        std[std == 0] = 1.0
        train_x = (train_x - mean) / std

        self.feature_means[position] = mean
        self.feature_stds[position] = std

        # Normalize targets if enabled
        if self.normalize_target:
            t_mean = train_y.mean().item()
            t_std = train_y.std().item()
            if t_std == 0:
                t_std = 1.0
            train_y = (train_y - t_mean) / t_std
            self.target_means[position] = t_mean
            self.target_stds[position] = t_std
        else:
            self.target_means[position] = 0.0
            self.target_stds[position] = 1.0

        if self.model_type == "svgp":
            self._train_svgp(position, train_x, train_y)
        else:
            self._train_exact(position, train_x, train_y)

    def _train_exact(self, position: Position, train_x: torch.Tensor, train_y: torch.Tensor) -> None:
        """Train an ExactGP model with subsampling."""
        if len(train_x) > MAX_TRAIN_SAMPLES:
            print(f"  Subsampling from {len(train_x)} to {MAX_TRAIN_SAMPLES} for tractability")
            indices = torch.randperm(len(train_x))[:MAX_TRAIN_SAMPLES]
            train_x = train_x[indices]
            train_y = train_y[indices]

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPRegressionModel(train_x, train_y, likelihood, kernel_name=self.kernel_name)

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

    def _train_svgp(self, position: Position, train_x: torch.Tensor, train_y: torch.Tensor) -> None:
        """Train a Stochastic Variational GP model with inducing points."""
        n_inducing = min(NUM_INDUCING_POINTS, len(train_x))
        print(f"  Using {n_inducing} inducing points on {len(train_x)} training examples")

        # Select inducing points via random subsampling
        inducing_indices = torch.randperm(len(train_x))[:n_inducing]
        inducing_points = train_x[inducing_indices].clone()

        model = SVGPModel(inducing_points, kernel_name=self.kernel_name)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=LEARNING_RATE)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_y))

        dataset = torch.utils.data.TensorDataset(train_x, train_y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(TRAINING_EPOCHS):
            epoch_loss = 0.0
            n_batches = 0
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{TRAINING_EPOCHS} - Loss: {epoch_loss / n_batches:.3f}")

        model.eval()
        likelihood.eval()

        self.models[position] = model
        self.likelihoods[position] = likelihood

    def _predict_position(self, position: Position, feature_path: Path) -> list[dict]:
        """Predict xP for all players at a given position."""
        model = self.models[position]
        likelihood = self.likelihoods[position]
        columns = self.feature_columns[position]
        feat_mean = self.feature_means[position]
        feat_std = self.feature_stds[position]
        t_mean = self.target_means[position]
        t_std = self.target_stds[position]

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
                x = (x - feat_mean) / feat_std

                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    pred_dist = likelihood(model(x))
                    pred_mean = pred_dist.mean.item()
                    variance = pred_dist.variance.item()

                # Denormalize prediction
                pred_mean = pred_mean * t_std + t_mean

                predictions.append(
                    {
                        "player_id": player_id,
                        "position": position.value,
                        "xp": round(pred_mean, 2),
                        "xp_uncertainty": round((variance**0.5) * t_std, 2),
                    }
                )

        return predictions
