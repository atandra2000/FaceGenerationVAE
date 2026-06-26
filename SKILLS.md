# SKILLS.md — FaceGenerationVAE

> Skills for the β-VAE face generation project.

---

## Skill 1: Run a baseline VAE training

```bash
cd Vision/FaceGenerationVAE
python src/train.py --epochs 50 --batch_size 64 --beta_max 1.0
```

CelebA is auto-downloaded on first run (~1.3 GB).

## Skill 2: Diagnose posterior collapse

**Symptom:** KL ≈ 0, samples look like the mean face.

1. **Verify KL annealing** is enabled and β reaches its target by epoch 30:
   ```python
   # src/train.py
   beta = min(beta_max, beta_max * epoch / 30)
   ```
2. **Check σ** — if σ ≈ 0 across dimensions, the model is collapsing.
3. **Lower free-bits** threshold from 0.5 nats to 0.1 nats per dim
   (encourages the model to actually use z).
4. **Add warm-up epochs** — start β_max at 0.001, not 0 (avoids
   dead-z at the start).

## Skill 3: Generate samples from the prior

```bash
python src/generate.py --checkpoint checkpoints/vae_epoch_49.pt \
  --n_samples 64 --output samples/epoch_49.png
```

The prior is N(0, I); sampling from it requires no encoder.

## Skill 4: Tune β (KL weight)

| β | Effect |
|---|--------|
| 0.1 | Blurry reconstructions, well-clustered latent |
| 1.0 | Standard β-VAE |
| 4.0 | Sharper, more disentangled (TC-VAE style) |
| 10.0 | Very sharp, but reconstructions get noisy |

The default `β_max=1.0` is the β-VAE sweet spot.

## Skill 5: Add a perceptual loss (instead of MSE)

Replace `F.mse_loss(x_hat, x)` with VGG-19 perceptual:

```python
import torchvision
vgg = torchvision.models.vgg19(weights="DEFAULT").features[:35].eval()
for p in vgg.parameters(): p.requires_grad = False

def perceptual_loss(x_hat, x):
    x_hat_vgg = vgg(x_hat)
    x_vgg = vgg(x)
    return F.mse_loss(x_hat_vgg, x_vgg)
```

**Trade-off:** perceptual loss produces sharper images but loses the
simple MSE-interpretable metric.

## Skill 6: Switch to a hierarchical VAE

Add a top-down stack of latent groups (Ladder VAE / NVAE):

```python
# src/model.py — HierarchicalVAE
z_groups = []
h = x
for block in self.encoder_blocks:
    h, z = block(h)
    z_groups.append(z)
```

Hierarchical VAEs scale to higher resolutions (256×256) without
posterior collapse. Significantly more compute.

## Pitfalls
- **MSE on [-1, 1] images** — fine for this project. For [0, 1] use
  `F.binary_cross_entropy_with_logits(x_logits, x)` instead.
- **Clamping log σ** to [-10, 10] prevents σ=0 but is a hard cutoff; a
  softplus floor (`σ = softplus(log_σ) + 1e-3`) is smoother.
- **`num_workers=4`** on P100 with batch 64 is the sweet spot. More workers
  don't help due to Kaggle I/O limits.
- **No BatchNorm in the encoder output** — it would re-normalize μ and σ.

