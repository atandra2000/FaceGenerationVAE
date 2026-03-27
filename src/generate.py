"""
Generation and interpolation — FaceGenerationVAE

Modes:
  sample       — sample N faces from the prior N(0, I)
  reconstruct  — encode + decode a set of real images
  interpolate  — smooth latent-space interpolation between two faces
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from model import VAE


def load_model(checkpoint: str, n_filters: int, latent_dim: int, device: torch.device) -> VAE:
    vae = VAE(in_channels=3, n_filters=n_filters, latent_dim=latent_dim).to(device)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    vae.load_state_dict(state)
    vae.eval()
    return vae


def load_image(path: str, image_size: int, device: torch.device) -> torch.Tensor:
    tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.CenterCrop(image_size),
        T.ToTensor(),
    ])
    img = Image.open(path).convert("RGB")
    return tf(img).unsqueeze(0).to(device)


def mode_sample(vae, args, device):
    """Sample `n_samples` faces from N(0, I) and save as a grid."""
    print(f"Sampling {args.n_samples} faces...")
    imgs = vae.sample(args.n_samples, device)
    grid = make_grid(imgs, nrow=int(args.n_samples ** 0.5), padding=2, normalize=False)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)
    print(f"Saved: {out_path}")


def mode_reconstruct(vae, args, device):
    """Encode and decode input images, saving side-by-side comparisons."""
    input_paths = sorted(Path(args.input_dir).glob("*.jpg")) + \
                  sorted(Path(args.input_dir).glob("*.png"))
    if not input_paths:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    originals, reconstructions = [], []
    for p in input_paths[:args.n_samples]:
        x = load_image(str(p), args.image_size, device)
        with torch.no_grad():
            recon, _, _ = vae(x)
        originals.append(x.squeeze(0))
        reconstructions.append(recon.squeeze(0))

    combined = []
    for o, r in zip(originals, reconstructions):
        combined.extend([o, r])
    grid = make_grid(combined, nrow=2, padding=2, normalize=False)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)
    print(f"Saved {len(originals)} reconstructions to {out_path}")


def mode_interpolate(vae, args, device):
    """Linearly interpolate in latent space between two images."""
    x1 = load_image(args.image1, args.image_size, device)
    x2 = load_image(args.image2, args.image_size, device)
    with torch.no_grad():
        frames = vae.interpolate(x1.squeeze(0), x2.squeeze(0), steps=args.steps)
    grid = make_grid(frames, nrow=args.steps, padding=2, normalize=False)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)
    print(f"Saved {args.steps}-step interpolation to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="VAE generation and interpolation")
    parser.add_argument("--checkpoint",  required=True,        help="Path to model checkpoint (.pth)")
    parser.add_argument("--mode",        default="sample",     choices=["sample", "reconstruct", "interpolate"])
    parser.add_argument("--output",      default="results/generated.png")
    parser.add_argument("--n-samples",   default=16,           type=int,  help="Images to generate/reconstruct")
    parser.add_argument("--steps",       default=10,           type=int,  help="Interpolation steps")
    parser.add_argument("--image-size",  default=128,          type=int)
    parser.add_argument("--n-filters",   default=64,           type=int)
    parser.add_argument("--latent-dim",  default=512,          type=int)
    parser.add_argument("--input-dir",   default="",           help="[reconstruct] Directory of input images")
    parser.add_argument("--image1",      default="",           help="[interpolate] First image path")
    parser.add_argument("--image2",      default="",           help="[interpolate] Second image path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_model(args.checkpoint, args.n_filters, args.latent_dim, device)
    print(f"Loaded VAE ({vae.num_parameters():,} params) on {device}")

    if args.mode == "sample":
        mode_sample(vae, args, device)
    elif args.mode == "reconstruct":
        mode_reconstruct(vae, args, device)
    elif args.mode == "interpolate":
        mode_interpolate(vae, args, device)


if __name__ == "__main__":
    main()
