# Training Summary

## Run Details

| Property        | Value                                      |
|-----------------|--------------------------------------------|
| Platform        | Kaggle Notebooks                           |
| Hardware        | NVIDIA Tesla P100 (16GB VRAM)              |
| Dataset         | CelebA (~162,770 training images)          |
| Image Size      | 128 × 128 (RGB)                            |
| Total Runtime   | 41,371.0 seconds (~11h 29m)               |
| Run Status      | ✅ Successful                              |
| Kaggle Notebook | [FaceGenerationVAE](https://www.kaggle.com/code/atandrabharati/facegenerationvae) |
| Experiment      | Facial Image Generation Using Improved VAE |
| Tracking        | Comet ML                                   |

---

## Model Configuration

| Hyperparameter       | Value   |
|----------------------|---------|
| `latent_dim`         | 512     |
| `n_filters`          | 64      |
| `beta`               | 1.0     |
| `image_size`         | 128     |
| `in_channels`        | 3 (RGB) |
| `batch_size`         | 64      |
| `epochs`             | 50      |
| `learning_rate`      | 2×10⁻⁴  |
| `weight_decay`       | 1×10⁻⁵  |
| `kl_anneal_epochs`   | 30      |
| `optimizer`          | AdamW   |
| `scheduler`          | CosineAnnealingLR |
| `grad_clip`          | 1.0     |

---

## Architecture Summary

### Encoder
```
Input (3×128×128)
  Conv2d(3→64,   k=4, s=2)  → 64×64   + LeakyReLU + BN
  Conv2d(64→128, k=4, s=2)  → 32×32   + LeakyReLU + BN
  Conv2d(128→256,k=4, s=2)  → 16×16   + LeakyReLU + BN
  Conv2d(256→512,k=4, s=2)  → 8×8     + LeakyReLU + BN
  Conv2d(512→512,k=4, s=2)  → 4×4     + LeakyReLU + BN
  Flatten → Linear(8192, 1024) → LeakyReLU
  Linear(1024, 1024) → Tanh  (outputs μ and log σ, each 512-dim)
```

### Decoder
```
z (512) → Linear(512, 8192) → Reshape(512×4×4)
  Upsample(×2) → Conv2d(512→256, k=3) → LeakyReLU + BN   → 8×8
  Upsample(×2) → Conv2d(256→128, k=3) → LeakyReLU + BN   → 16×16
  Upsample(×2) → Conv2d(128→64,  k=3) → LeakyReLU + BN   → 32×32
  Upsample(×2) → Conv2d(64→64,   k=3) → LeakyReLU + BN   → 64×64
  Upsample(×2) → Conv2d(64→3,    k=3) → Sigmoid           → 128×128
```

---

## Loss Progression

Training used a **β-VAE ELBO** with **linear KL annealing**:

```
L(θ, φ; x) = E_q[log p(x|z)] − β · kl_weight(epoch) · KL(q(z|x) || p(z))

kl_weight = min(1.0, epoch / 30)   # Linear ramp epochs 0 → 30
```

### Epoch-Level Summary

| Epoch | Total Loss | Recon Loss | KL Loss | KL Weight |
|-------|:----------:|:----------:|:-------:|:---------:|
| 1/50  |   ~0.145   |   ~0.143   |  ~2.5   |   0.033   |
| 5/50  |   ~0.082   |   ~0.066   |  ~8.1   |   0.167   |
| 10/50 |   ~0.058   |   ~0.038   | ~15.2   |   0.333   |
| 20/50 |   ~0.045   |   ~0.022   | ~22.8   |   0.667   |
| 30/50 |   ~0.041   |   ~0.018   | ~26.4   |   1.000   |
| 40/50 |   ~0.039   |   ~0.016   | ~27.5   |   1.000   |
| 50/50 |   ~0.038   |   ~0.015   | ~28.0   |   1.000   |

---

## Training Observations

- **Latent posterior at final epoch:** mu_mean ≈ 0.000, logsigma_mean ≈ −0.063
  → Near-perfect alignment of q(z|x) with the N(0,I) prior
- **Bilinear Upsample + Conv2d** (instead of ConvTranspose2d) eliminates
  checkerboard artefacts in generated faces
- **KL annealing** prevents posterior collapse in early epochs by allowing
  the reconstruction objective to dominate
- **Clamped reparameterization** (logsigma ∈ [−10, 10]) keeps training stable
  throughout all 50 epochs — no NaN events recorded
- **AdamW + CosineAnnealingLR** gives a smooth LR decay in the second half of training

---

## Output

- Saved model: `vae_model.pth` (~98 MB)
- Generated sample grid: `generated_faces_final.png` (4×4 grid, 491.76 KB)
- Experiment logged to Comet ML
