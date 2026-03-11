"""
Unit tests for LaneNet model.

Tests forward pass, loss computation, and inference.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import create_lanenet, LaneNet


class TestLaneNetModel:
    """Test LaneNet model."""

    @pytest.fixture
    def device(self):
        """Get device for testing."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.fixture
    def model(self, device):
        """Create model for testing."""
        model = create_lanenet(backbone='efficientnet', pretrained=False)
        return model.to(device)

    def test_model_creation(self, model):
        """Test model can be created."""
        assert model is not None
        assert isinstance(model, LaneNet)

    def test_model_parameters(self, model):
        """Test model has trainable parameters."""
        params = list(model.parameters())
        assert len(params) > 0
        assert any(p.requires_grad for p in params)

    def test_forward_pass_single(self, model, device):
        """Test forward pass with single image."""
        batch_size = 1
        x = torch.randn(batch_size, 3, 384, 640, device=device)

        with torch.no_grad():
            outputs = model(x)

        assert 'seg' in outputs
        assert 'emb' in outputs
        assert outputs['seg'].shape == (batch_size, 1, 384, 640)
        assert outputs['emb'].shape == (batch_size, 4, 384, 640)

    def test_forward_pass_batch(self, model, device):
        """Test forward pass with batch."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 384, 640, device=device)

        with torch.no_grad():
            outputs = model(x)

        assert outputs['seg'].shape == (batch_size, 1, 384, 640)
        assert outputs['emb'].shape == (batch_size, 4, 384, 640)

    def test_forward_pass_different_sizes(self, model, device):
        """Test forward pass with different input sizes."""
        sizes = [(192, 320), (256, 384), (480, 640)]

        for h, w in sizes:
            x = torch.randn(1, 3, h, w, device=device)

            with torch.no_grad():
                outputs = model(x)

            assert outputs['seg'].shape == (1, 1, h, w)
            assert outputs['emb'].shape == (1, 4, h, w)

    def test_forward_pass_gradients(self, model, device):
        """Test gradients flow through model."""
        x = torch.randn(2, 3, 384, 640, device=device, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 384, 640), device=device, dtype=torch.float32)

        outputs = model(x)
        loss = torch.nn.BCEWithLogitsLoss()(outputs['seg'], target)
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_model_eval_mode(self, model):
        """Test model can switch to eval mode."""
        model.eval()
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                assert not module.training

    def test_model_train_mode(self, model):
        """Test model can switch to train mode."""
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                assert module.training

    def test_model_inference_deterministic(self, model, device):
        """Test inference produces same outputs with same input."""
        model.eval()
        x = torch.randn(1, 3, 384, 640, device=device)

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert torch.allclose(out1['seg'], out2['seg'])
        assert torch.allclose(out1['emb'], out2['emb'])

    def test_loss_computation(self, model, device):
        """Test loss computation."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 384, 640, device=device)
        target = torch.randint(0, 2, (batch_size, 1, 384, 640), device=device, dtype=torch.float32)

        outputs = model(x)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(outputs['seg'], target)

        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_backbone_options(self, device):
        """Test different backbone options."""
        for backbone in ['efficientnet', 'mobilenet']:
            model = create_lanenet(backbone=backbone, pretrained=False)
            model = model.to(device)

            x = torch.randn(1, 3, 384, 640, device=device)
            with torch.no_grad():
                outputs = model(x)

            assert outputs['seg'].shape == (1, 1, 384, 640)
            assert outputs['emb'].shape == (1, 4, 384, 640)

    def test_embedding_dimension(self, device):
        """Test different embedding dimensions."""
        for emb_dim in [2, 4, 8, 16]:
            model = create_lanenet(
                backbone='efficientnet',
                pretrained=False,
                embedding_dim=emb_dim
            )
            model = model.to(device)

            x = torch.randn(1, 3, 384, 640, device=device)
            with torch.no_grad():
                outputs = model(x)

            assert outputs['emb'].shape == (1, emb_dim, 384, 640)

    def test_model_parameter_count(self, model):
        """Test model parameter count."""
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        assert total_params < 50_000_000  # Should be reasonable size

    def test_model_memory_usage(self, model, device):
        """Test model memory usage."""
        torch.cuda.empty_cache() if device.type == 'cuda' else None

        x = torch.randn(1, 3, 384, 640, device=device)

        with torch.no_grad():
            outputs = model(x)

        if device.type == 'cuda':
            memory_used = torch.cuda.memory_allocated() / (1024 ** 2)
            assert memory_used < 2000  # Less than 2GB for single image

    def test_model_inference_speed(self, model, device):
        """Test model inference speed."""
        model.eval()
        x = torch.randn(1, 3, 384, 640, device=device)

        import time
        with torch.no_grad():
            start = time.perf_counter()
            for _ in range(10):
                outputs = model(x)
            elapsed = (time.perf_counter() - start) / 10 * 1000  # ms

        assert elapsed < 100  # Should be reasonably fast

    def test_model_print_summary(self, model, capsys):
        """Test model summary printing."""
        model.print_summary()
        captured = capsys.readouterr()
        assert 'Parameters' in captured.out
        assert 'Model Size' in captured.out


class TestLaneNetIntegration:
    """Integration tests for LaneNet."""

    def test_end_to_end_pipeline(self):
        """Test complete training pipeline."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model
        model = create_lanenet(backbone='efficientnet', pretrained=False)
        model = model.to(device)

        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        # Simulate training loop
        for _ in range(3):  # 3 iterations
            # Forward
            x = torch.randn(2, 3, 384, 640, device=device)
            target = torch.randint(0, 2, (2, 1, 384, 640), device=device, dtype=torch.float32)

            outputs = model(x)
            loss = loss_fn(outputs['seg'], target)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            assert not torch.isnan(loss)

    def test_inference_on_random_data(self):
        """Test inference on random data."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = create_lanenet(backbone='efficientnet', pretrained=False)
        model.eval()
        model = model.to(device)

        # Test multiple batches
        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 3, 384, 640, device=device)

            with torch.no_grad():
                outputs = model(x)

            assert outputs['seg'].shape == (batch_size, 1, 384, 640)
            assert outputs['emb'].shape == (batch_size, 4, 384, 640)


class TestLaneNetUtils:
    """Test utility functions."""

    def test_create_lanenet_factory(self):
        """Test factory function."""
        model = create_lanenet(
            num_classes=1,
            backbone='efficientnet',
            pretrained=False,
            embedding_dim=4
        )
        assert isinstance(model, LaneNet)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
