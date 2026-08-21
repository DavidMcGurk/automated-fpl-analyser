"""Tests for GP model configuration and validation."""

from unittest.mock import patch

import torch

from src.optimisation.gp_model import (
    GPModel,
    GPRegressionModel,
    SVGPModel,
    _make_base_kernel,
)
from src.models.post_prediction import Position
import gpytorch


class TestMakeBaseKernel:
    def test_rbf_kernel(self):
        kernel = _make_base_kernel("rbf", 5)
        assert isinstance(kernel, gpytorch.kernels.RBFKernel)
        assert kernel.ard_num_dims == 5

    def test_matern32_kernel(self):
        kernel = _make_base_kernel("matern32", 3)
        assert isinstance(kernel, gpytorch.kernels.MaternKernel)
        assert kernel.nu == 1.5

    def test_matern52_kernel(self):
        kernel = _make_base_kernel("matern52", 4)
        assert isinstance(kernel, gpytorch.kernels.MaternKernel)
        assert kernel.nu == 2.5

    def test_unknown_kernel_raises(self):
        try:
            _make_base_kernel("polynomial", 2)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestGPModelConfig:
    def test_default_config(self):
        model = GPModel()
        assert model.model_type == "exact"
        assert model.kernel_name == "rbf"
        assert model.normalize_target is True

    def test_svgp_config(self):
        model = GPModel(model_type="svgp", kernel_name="matern32")
        assert model.model_type == "svgp"
        assert model.kernel_name == "matern32"

    def test_no_normalize_target(self):
        model = GPModel(normalize_target=False)
        assert model.normalize_target is False


class TestGPRegressionModel:
    def test_rbf_model_creation(self):
        train_x = torch.randn(10, 5)
        train_y = torch.randn(10)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPRegressionModel(train_x, train_y, likelihood, kernel_name="rbf")
        assert isinstance(model.covar_module.base_kernel, gpytorch.kernels.RBFKernel)

    def test_matern52_model_creation(self):
        train_x = torch.randn(10, 5)
        train_y = torch.randn(10)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPRegressionModel(train_x, train_y, likelihood, kernel_name="matern52")
        assert isinstance(model.covar_module.base_kernel, gpytorch.kernels.MaternKernel)
        assert model.covar_module.base_kernel.nu == 2.5


class TestSVGPModel:
    def test_svgp_creation(self):
        inducing_points = torch.randn(20, 5)
        model = SVGPModel(inducing_points, kernel_name="matern32")
        assert isinstance(model.covar_module.base_kernel, gpytorch.kernels.MaternKernel)
        assert model.covar_module.base_kernel.nu == 1.5

    def test_svgp_forward(self):
        inducing_points = torch.randn(10, 3)
        model = SVGPModel(inducing_points, kernel_name="rbf")
        model.eval()
        x = torch.randn(5, 3)
        with torch.no_grad():
            output = model(x)
        assert output.mean.shape == (5,)


class TestTargetNormalization:
    @patch.object(GPModel, "_train_exact")
    @patch.object(GPModel, "_train_svgp")
    def test_normalize_target_stores_stats(self, mock_svgp, mock_exact):
        """When normalize_target=True, target mean/std should be stored."""
        model = GPModel(model_type="exact", normalize_target=True)
        train_x = torch.randn(20, 5)
        train_y = torch.tensor([1.0, 2.0, 3.0] * 6 + [1.0, 2.0], dtype=torch.float32)

        model._train_position(Position.GOALKEEPER, train_x, train_y)

        assert Position.GOALKEEPER in model.target_means
        assert Position.GOALKEEPER in model.target_stds
        assert model.target_stds[Position.GOALKEEPER] > 0

    @patch.object(GPModel, "_train_exact")
    def test_no_normalize_target_stores_identity(self, mock_exact):
        """When normalize_target=False, target stats should be identity (0, 1)."""
        model = GPModel(model_type="exact", normalize_target=False)
        train_x = torch.randn(20, 5)
        train_y = torch.randn(20)

        model._train_position(Position.GOALKEEPER, train_x, train_y)

        assert model.target_means[Position.GOALKEEPER] == 0.0
        assert model.target_stds[Position.GOALKEEPER] == 1.0

    @patch.object(GPModel, "_train_exact")
    def test_feature_standardization_always_applied(self, mock_exact):
        """Feature standardization should always happen regardless of target normalization."""
        model = GPModel(normalize_target=False)
        train_x = torch.randn(20, 5) * 10 + 5  # Non-standardized
        train_y = torch.randn(20)

        model._train_position(Position.DEFENDER, train_x, train_y)

        feat_mean = model.feature_means[Position.DEFENDER]
        feat_std = model.feature_stds[Position.DEFENDER]
        assert feat_mean.shape == (5,)
        assert feat_std.shape == (5,)


class TestValidate:
    def test_validate_returns_metrics(self):
        """validate() should return RMSE, MAE, and means per position."""
        model = GPModel(model_type="exact", normalize_target=True)

        # Mock _load_training_data to return small tensors
        def mock_load(position, seasons):
            x = torch.randn(30, 5)
            y = torch.randn(30) * 3 + 2
            cols = ["a", "b", "c", "d", "e"]
            return x, y, cols

        with patch.object(GPModel, "_load_training_data", side_effect=mock_load):
            results = model.validate(["2023_24"], ["2024_25"])

        assert len(results) == 4  # All 4 positions
        for position, metrics in results.items():
            assert "rmse" in metrics
            assert "mae" in metrics
            assert "mean_pred" in metrics
            assert "mean_actual" in metrics
            assert metrics["rmse"] >= 0
            assert metrics["mae"] >= 0
