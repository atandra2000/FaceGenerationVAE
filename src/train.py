"""
Training loop — FaceGenerationVAE

Full training pipeline with:
  - β-VAE ELBO objective with linear KL annealing
  - AdamW + CosineAnnealingLR
  - NaN detection guard
  - Comet ML experiment tracking
  - Per-epoch checkpoint saving
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from model import VAE, vae_loss
from dataset import build_dataloaders

try:
    import comet_ml
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False


def kl_annealing_weight(epoch: int, kl_anneal_epochs: int) -> float:
    """Linear KL weight ramp: 0 at epoch 0 → 1 at kl_anneal_epochs."""
    return min(1.0, epoch / max(kl_anneal_epochs, 1))


def create_comet_experiment(project_name: str, params: dict):
    """Initialise a Comet ML experiment and log all hyperparameters."""
    if not COMET_AVAILABLE:
        return None
    api_key = os.environ.get("COMET_API_KEY", "")
    if not api_key:
        print("COMET_API_KEY not set — skipping Comet ML tracking.")
        return None
    experiment = comet_ml.Experiment(api_key=api_key, project_name=project_name)
    for k, v in params.items():
        experiment.log_parameter(k, v)
    experiment.flush()
    return experiment


def train_one_epoch(vae, loader, optimizer, device, epoch, beta, kl_anneal_epochs):
    """Run one training epoch. Returns (avg_total, avg_recon, avg_kl)."""
    vae.train()
    kl_weight = kl_annealing_weight(epoch, kl_anneal_epochs)
    total_loss = recon_loss_sum = kl_loss_sum = 0.0

    for x, _ in tqdm(loader, desc="Training", leave=False):
        x = x.to(device)
        optimizer.zero_grad()

        x_recon, mu, logsigma = vae(x)

        if torch.isnan(mu).any() or torch.isnan(logsigma).any():
            print(f"  [Epoch {epoch}] Warning: NaN in mu/logsigma — skipping batch")
            continue

        loss, r_loss, k_loss = vae_loss(x, x_recon, mu, logsigma, beta, kl_weight)

        if torch.isnan(loss):
            print(f"  [Epoch {epoch}] Warning: NaN loss — skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss     += loss.item()
        recon_loss_sum += r_loss.item()
        kl_loss_sum    += k_loss.item()

    n = len(loader)
    return total_loss / n, recon_loss_sum / n, kl_loss_sum / n


@torch.no_grad()
def validate(vae, loader, device, epoch, beta, kl_anneal_epochs):
    """Run validation loop. Returns (avg_total, avg_recon, avg_kl)."""
    vae.eval()
    kl_weight = kl_annealing_weight(epoch, kl_anneal_epochs)
    total_loss = recon_loss_sum = kl_loss_sum = 0.0

    for x, _ in tqdm(loader, desc="Validation", leave=False):
        x = x.to(device)
        x_recon, mu, logsigma = vae(x)
        loss, r_loss, k_loss = vae_loss(x, x_recon, mu, logsigma, beta, kl_weight)
        total_loss     += loss.item()
        recon_loss_sum += r_loss.item()
        kl_loss_sum    += k_loss.item()

    n = len(loader)
    return total_loss / n, recon_loss_sum / n, kl_loss_sum / n


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, in_channels = build_dataloaders(
        data_root=cfg.data_root,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        download=True,
    )
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    vae = VAE(in_channels=in_channels, n_filters=cfg.n_filters, latent_dim=cfg.latent_dim).to(device)
    print(f"VAE parameters: {vae.num_parameters():,}")

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = optim.AdamW(vae.parameters(), lr=cfg.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # ── Comet ML ──────────────────────────────────────────────────────────────
    params = {
        "batch_size": cfg.batch_size, "epochs": cfg.epochs,
        "learning_rate": cfg.lr, "latent_dim": cfg.latent_dim,
        "beta": cfg.beta, "image_size": cfg.image_size,
        "n_filters": cfg.n_filters, "kl_anneal_epochs": cfg.kl_anneal_epochs,
    }
    experiment = create_comet_experiment("Facial Image Generation Using Improved VAE", params)

    # ── Training loop ─────────────────────────────────────────────────────────
    save_dir = Path(cfg.checkpoint_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(cfg.epochs):
        kl_w = kl_annealing_weight(epoch, cfg.kl_anneal_epochs)
        tr_loss, tr_recon, tr_kl = train_one_epoch(
            vae, train_loader, optimizer, device, epoch, cfg.beta, cfg.kl_anneal_epochs
        )
        val_loss, val_recon, val_kl = validate(
            vae, val_loader, device, epoch, cfg.beta, cfg.kl_anneal_epochs
        )
        scheduler.step()

        print(
            f"Epoch [{epoch+1:02d}/{cfg.epochs}]  "
            f"Train Loss={tr_loss:.4f}  Recon={tr_recon:.4f}  KL={tr_kl:.2f}  "
            f"KL-w={kl_w:.3f}  |  "
            f"Val Loss={val_loss:.4f}  Recon={val_recon:.4f}  KL={val_kl:.2f}"
        )

        if experiment:
            experiment.log_metrics({
                "train_loss": tr_loss, "train_recon_loss": tr_recon, "train_kl_loss": tr_kl,
                "val_loss": val_loss, "val_recon_loss": val_recon, "val_kl_loss": val_kl,
                "kl_weight": kl_w,
            }, epoch=epoch)

        # Save checkpoint
        ckpt_path = save_dir / f"checkpoint_epoch_{epoch+1:02d}.pth"
        torch.save({
            "epoch": epoch + 1,
            "model_state": vae.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
        }, ckpt_path)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(vae.state_dict(), save_dir / "best_model.pth")
            print(f"  ✓ New best val loss: {best_val:.4f}")

    if experiment:
        experiment.end()
    print(f"\nTraining complete. Best val loss: {best_val:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train VAE on CelebA")
    parser.add_argument("--data-root",       default="celeba-data",   type=str)
    parser.add_argument("--checkpoint-dir",  default="checkpoints",   type=str)
    parser.add_argument("--epochs",          default=50,              type=int)
    parser.add_argument("--batch-size",      default=64,              type=int)
    parser.add_argument("--lr",              default=2e-4,            type=float)
    parser.add_argument("--latent-dim",      default=512,             type=int)
    parser.add_argument("--beta",            default=1.0,             type=float)
    parser.add_argument("--image-size",      default=128,             type=int)
    parser.add_argument("--n-filters",       default=64,              type=int)
    parser.add_argument("--kl-anneal-epochs",default=30,              type=int)
    parser.add_argument("--num-workers",     default=4,               type=int)
    args = parser.parse_args()

    # Map hyphenated args to underscored cfg attributes
    args.kl_anneal_epochs = args.kl_anneal_epochs
    args.data_root        = args.data_root
    args.checkpoint_dir   = args.checkpoint_dir
    args.n_filters        = args.n_filters
    args.latent_dim       = args.latent_dim
    args.num_workers      = args.num_workers
    args.image_size       = args.image_size

    train(args)


if __name__ == "__main__":
    main()
