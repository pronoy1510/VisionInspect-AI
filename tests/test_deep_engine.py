"""Unit tests for DeepAnomalyEngine."""
import numpy as np
import torch
import pytest
from vision_inspect.deep_engine import DeepAnomalyEngine, ConvAnomalyAutoencoder


def test_autoencoder_tensor_shapes():
    model = ConvAnomalyAutoencoder(in_channels=1, latent_dim=64)
    x = torch.randn(2, 1, 256, 256)
    out = model(x)
    assert out.shape == (2, 1, 256, 256)
    # Output should be in [0, 1] due to sigmoid
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_deep_anomaly_engine_inference():
    engine = DeepAnomalyEngine()
    dummy = np.random.uniform(0.1, 0.9, (256, 256)).astype(np.float32)
    heat, score = engine.compute_anomaly_map(dummy)

    assert heat.shape == (256, 256)
    assert 0.0 <= heat.min() <= heat.max() <= 1.0
    assert isinstance(score, float)
    assert score >= 0.0
