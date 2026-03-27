"""
Hyperparameter configuration — FaceGenerationVAE

Single source of truth for all training and architecture settings.
All values match the original Kaggle P100 training run exactly.
"""

from dataclasses import dataclass, field


@dataclass
class VAEConfig:
    # ── Data ──────────────────────────────────────────────────────────────────
    data_root:    str = "celeba-data"    # Directory where CelebA is stored
    image_size:   int = 128             # Spatial resolution (H = W)
    in_channels:  int = 3               # RGB

    # ── Architecture ──────────────────────────────────────────────────────────
    latent_dim:   int = 512             # Dimensionality of z
    n_filters:    int = 64              # Base conv filter count (doubles per layer)
    beta:       float = 1.0             # β coefficient on KL term (β-VAE)

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size:         int = 64
    epochs:             int = 50
    lr:               float = 2e-4      # AdamW learning rate
    weight_decay:     float = 1e-5      # AdamW weight decay
    kl_anneal_epochs:   int = 30        # Linear KL ramp: 0 → 1 over this many epochs
    grad_clip:        float = 1.0       # Max gradient norm

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer:  str = "AdamW"
    scheduler:  str = "CosineAnnealingLR"

    # ── Hardware ──────────────────────────────────────────────────────────────
    num_workers:  int = 4
    pin_memory:  bool = True

    # ── Paths ─────────────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    results_dir:    str = "results"

    # ── Comet ML ──────────────────────────────────────────────────────────────
    project_name: str = "Facial Image Generation Using Improved VAE"


# Default config instance
cfg = VAEConfig()


if __name__ == "__main__":
    from dataclasses import asdict
    import json
    print(json.dumps(asdict(cfg), indent=2))
