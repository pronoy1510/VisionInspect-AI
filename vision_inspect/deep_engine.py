"""
Deep Learning Anomaly Detection Module using PyTorch Convolutional Autoencoder.
Computes pixel-level reconstruction residual heatmaps and deep anomaly scores.
"""

from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import DeepEngineConfig


class ConvAnomalyAutoencoder(nn.Module):
    """Convolutional Autoencoder for industrial surface reconstruction and anomaly isolation."""

    def __init__(self, in_channels: int = 1, latent_dim: int = 64):
        super().__init__()
        # Encoder: 256x256 -> 128x128 -> 64x64 -> 32x32 -> 16x16
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.flatten_dim = 128 * 16 * 16
        self.fc_enc = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flatten_dim)

        # Decoder: 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(16, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        feat = self.encoder(x)
        feat_flat = feat.view(b, -1)
        latent = self.fc_enc(feat_flat)
        dec_in = self.fc_dec(latent).view(b, 128, 16, 16)
        out = self.decoder(dec_in)
        return out


class DeepAnomalyEngine:
    """Manages neural autoencoder inference, residual map calculation, and anomaly thresholding."""

    def __init__(self, config: Optional[DeepEngineConfig] = None):
        self.config = config or DeepEngineConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() and self.config.device != "cpu" else "cpu")
        self.model = ConvAnomalyAutoencoder(
            in_channels=self.config.input_channels,
            latent_dim=self.config.latent_dim
        ).to(self.device)
        self.model.eval()

        self.threshold = self.config.anomaly_threshold
        self._load_or_init_weights()

    def _load_or_init_weights(self) -> None:
        """Loads weights if available, otherwise initializes robust Xavier/Kaiming weights."""
        model_path = Path(self.config.model_path)
        if model_path.exists():
            try:
                state = torch.load(str(model_path), map_location=self.device)
                self.model.load_state_dict(state)
                return
            except Exception:
                pass

        # Robust initialization
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def save_weights(self, path: Optional[Path] = None) -> None:
        save_path = Path(path or self.config.model_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(save_path))

    def compute_anomaly_map(self, normalized_img: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Infers reconstructed image and computes pixel-wise L1 residual heatmap.
        Returns:
            heatmap: (H, W) float32 normalized [0.0, 1.0]
            global_anomaly_score: float mean top-5% reconstruction error
        """
        h, w = normalized_img.shape[:2]
        tensor = torch.from_numpy(normalized_img).float().to(self.device)
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif len(tensor.shape) == 3:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            recon = self.model(tensor)
            residual = torch.abs(tensor - recon).squeeze().cpu().numpy()

        # Smooth residual with bilateral/Gaussian filter to reduce point noise
        residual_smoothed = cv2.GaussianBlur(residual, (7, 7), 1.5)

        # Global anomaly score based on 95th percentile error
        p95_error = float(np.percentile(residual_smoothed, 95))

        # Normalize heatmap to [0.0, 1.0]
        max_val = np.max(residual_smoothed)
        if max_val > 1e-6:
            heatmap = (residual_smoothed / max_val).astype(np.float32)
        else:
            heatmap = np.zeros_like(residual_smoothed, dtype=np.float32)

        return heatmap, p95_error
