# AGENTS.md — FaceGenerationVAE

> **Project:** `Vision/FaceGenerationVAE/` · **Type:** β-VAE
> **Task:** 128×128 face generation from 512-dim latent
> **Hardware:** NVIDIA P100 16GB · **Status:** complete (50 epochs)

Improved β-VAE for photorealistic face generation at 128×128 from a 512-dim
latent space, trained from scratch on CelebA. Generates novel faces by
sampling from the prior.

---

## 1. Subagent: `vae-debugger`

**Trigger:** "Posterior collapse", "KL annealing", "VAE samples are blurry",
"ELBO not decreasing", "How to fix KL=0?", "Add β-VAE warmup."

**System prompt:**
You are a senior VAE engineer. The headline challenge in β-VAE training is
**posterior collapse** — the decoder learns to ignore z. The cure is **linear
KL annealing** (β: 0 → 1 over the first 30 epochs).

**Architecture:**
- **Encoder:** 5 strided Conv2d (3→64→128→256→512→512) + 2 Linear → μ (512)
  / log σ (512). Tanh-bounded bottleneck, clamped to [-10, 10] for stability.
- **Decoder:** Linear → reshape + 5 (Upsample bilinear + Conv2d 3×3)
  stages — avoids checkerboard artefacts vs ConvTranspose2d.

**Training:**
- β-VAE ELBO with **linear KL annealing (0→1 over 30 epochs)**.
- MSE reconstruction, AdamW + CosineAnnealingLR, gradient clip 1.0.
- Comet ML tracking.

**Dataset:** CelebA (~162K images, auto-download ~1.3GB), 128×128,
HFlip/Rotation/ColorJitter.

**Results (epoch 50):** train MSE 0.0152, val MSE 0.0162, KL 28.0,
posterior μ ≈ 0, ~11h29m on P100.

**Files:**
- `configs/config.py` — `VAEConfig`.
- `src/{model,dataset,train,generate,utils}.py`.
- `results/training_summary.md`.
- `assets/training_curves.png`.

**Hard rules:**
1. **Never** use ConvTranspose2d in the decoder — it produces
   checkerboard artefacts. Use Upsample(bilinear) + Conv2d(3×3).
2. **Never** skip the KL annealing — without it, posterior collapses
   within the first 5 epochs.
3. **Always** clamp log σ to [-10, 10] (or use a Tanh-bounded bottleneck).
   Without it, the model can drive σ → 0 and the KL term → -∞.
4. **Always** verify μ ≈ 0 across dimensions before declaring training
   successful — it's the sign the prior is matching the aggregate posterior.

