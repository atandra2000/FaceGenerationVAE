"""
VAE Model — FaceGenerationVAE
Convolutional β-VAE for 128×128 face image generation.

Architecture:
  Encoder: 5× Conv2d(stride=2) → Flatten → 2 Linear layers → (μ, log σ)
  Decoder: Linear → Reshape → 4× Upsample+Conv2d (bilinear, no checkerboard)
  Reparameterization: clamped logsigma for numerical stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Encoder ───────────────────────────────────────────────────────────────────
def build_encoder(in_channels: int, n_filters: int, n_outputs: int) -> nn.Sequential:
    """
    Convolutional encoder: image → (μ, log σ) concatenated.

    Spatial reduction: 128 → 64 → 32 → 16 → 8 → 4
    Final feature map: (8×n_filters, 4, 4)

    Args:
        in_channels: Number of input image channels (3 for RGB).
        n_filters:   Base filter count (doubles per layer up to 8×).
        n_outputs:   Output size = 2 × latent_dim (μ and log σ stacked).
    """
    nf = n_filters
    model = nn.Sequential(
        # 128 → 64
        nn.Conv2d(in_channels, nf, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.BatchNorm2d(nf),
        # 64 → 32
        nn.Conv2d(nf, 2 * nf, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.BatchNorm2d(2 * nf),
        # 32 → 16
        nn.Conv2d(2 * nf, 4 * nf, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.BatchNorm2d(4 * nf),
        # 16 → 8
        nn.Conv2d(4 * nf, 8 * nf, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.BatchNorm2d(8 * nf),
        # 8 → 4
        nn.Conv2d(8 * nf, 8 * nf, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.BatchNorm2d(8 * nf),
        # Flatten → dense
        nn.Flatten(),
        nn.Linear(4 * 4 * 8 * nf, 1024),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(1024, n_outputs),
        nn.Tanh(),  # bound μ and log σ to [-1, 1] for stability
    )
    # Weight initialisation
    nn.init.kaiming_normal_(model[-3].weight, mode="fan_out", nonlinearity="leaky_relu")
    nn.init.xavier_normal_(model[-2].weight)
    return model


# ── Decoder ───────────────────────────────────────────────────────────────────
def build_decoder(n_filters: int, latent_dim: int, out_channels: int = 3) -> nn.Module:
    """
    Convolutional decoder: latent vector → 128×128 image.

    Uses bilinear Upsample + Conv2d instead of ConvTranspose2d to avoid
    checkerboard artefacts.

    Spatial expansion: 4 → 8 → 16 → 32 → 64 → 128
    """
    nf = n_filters

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(latent_dim, 4 * 4 * 8 * nf)
            self.nf = nf
            self.body = nn.Sequential(
                # 4 → 8
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(8 * nf, 4 * nf, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(4 * nf),
                # 8 → 16
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(4 * nf, 2 * nf, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(2 * nf),
                # 16 → 32
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(2 * nf, nf, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(nf),
                # 32 → 64
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(nf),
                # 64 → 128
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(nf, out_channels, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid(),
            )

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            x = self.fc(z)
            x = x.view(-1, 8 * self.nf, 4, 4)
            return self.body(x)

    return Decoder()


# ── VAE ───────────────────────────────────────────────────────────────────────
class VAE(nn.Module):
    """
    Variational Autoencoder for 128×128 RGB face generation.

    Implements the β-VAE objective:
        L = E[log p(x|z)] − β · KL(q(z|x) || p(z))

    With KL annealing: β is linearly ramped from 0 → 1 over kl_anneal_epochs.
    """

    def __init__(self, in_channels: int, n_filters: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = build_encoder(in_channels, n_filters, n_outputs=2 * latent_dim)
        self.decoder = build_decoder(n_filters, latent_dim, out_channels=in_channels)

    def encode(self, x: torch.Tensor):
        """Return (μ, log σ) split from encoder output."""
        out = self.encoder(x)
        mu, logsigma = out.chunk(2, dim=1)
        return mu, logsigma

    def reparameterize(self, mu: torch.Tensor, logsigma: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick with clamping for numerical stability.

        z = μ + σ · ε,  ε ~ N(0, I)
        """
        logsigma = torch.clamp(logsigma, -10, 10)
        eps = torch.randn_like(logsigma)
        return mu + torch.exp(logsigma) * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """Returns (reconstruction, μ, log σ)."""
        mu, logsigma = self.encode(x)
        mu = torch.clamp(mu, -10, 10)
        logsigma = torch.clamp(logsigma, -10, 10)
        z = self.reparameterize(mu, logsigma)
        recon = self.decode(z)
        return recon, mu, logsigma

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample n images by drawing z ~ N(0, I)."""
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)

    @torch.no_grad()
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, steps: int = 8) -> torch.Tensor:
        """
        Linearly interpolate in latent space between two images.

        Returns a tensor of shape (steps, C, H, W).
        """
        mu1, _ = self.encode(x1.unsqueeze(0))
        mu2, _ = self.encode(x2.unsqueeze(0))
        alphas = torch.linspace(0, 1, steps, device=x1.device)
        zs = torch.stack([(1 - a) * mu1 + a * mu2 for a in alphas]).squeeze(1)
        return self.decode(zs)


# ── Loss ──────────────────────────────────────────────────────────────────────
def vae_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    logsigma: torch.Tensor,
    beta: float,
    kl_weight: float,
):
    """
    β-VAE ELBO loss.

    Args:
        x:          Original images  [B, C, H, W]
        x_recon:    Reconstructions  [B, C, H, W]
        mu:         Posterior mean    [B, latent_dim]
        logsigma:   Posterior log σ  [B, latent_dim]
        beta:       β coefficient for KL term (1.0 = standard VAE)
        kl_weight:  Linear annealing weight 0 → 1

    Returns:
        (total_loss, recon_loss, kl_loss) — all scalars
    """
    recon_loss = F.mse_loss(x_recon, x, reduction="none").mean(dim=[1, 2, 3])
    kl_loss = 0.5 * torch.sum(torch.exp(logsigma) + mu ** 2 - 1 - logsigma, dim=1)
    total = (recon_loss + beta * kl_weight * kl_loss).mean()
    return total, recon_loss.mean(), kl_loss.mean()
