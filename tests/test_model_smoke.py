"""Smoke test for FaceGenerationVAE."""
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import VAE


def test_vae_shapes():
    batch_size = 2
    latent_dim = 64
    vae = VAE(in_channels=3, n_filters=16, latent_dim=latent_dim)
    vae.eval()

    x = torch.rand(batch_size, 3, 128, 128)
    recon, mu, logsigma = vae(x)

    assert recon.shape == (batch_size, 3, 128, 128)
    assert mu.shape == (batch_size, latent_dim)
    assert logsigma.shape == (batch_size, latent_dim)
    assert recon.min() >= 0.0 and recon.max() <= 1.0
