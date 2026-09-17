# Dual-Head TLSTM

A shared-encoder, two-head recurrent model for log realized volatility on a cross-section of assets. **Single-head is the shipped model. Dual-head is a research program with an explicit stop condition.**

Every input is $\mathcal{F}_t$-measurable, every split is chronological by calendar date and embargoed, and every structural claim in the design carries a proof. See [`MATH.md`](MATH.md).

---

## Why the v3.x dual-head is dead — a theorem, not a bug

For any encoder $\psi$, and targets $Z_A$ (primary) and $Z_B$ (auxiliary) with $Z_B = g(Z_A, X)$:

$$
I(Z_B; \psi) \;\le\; I(Z_A; \psi) + I(X; \psi).
$$

The primary gradient already maximizes $I(Z_A;\psi)$; Head B's gradient adds nothing that is not already accounted for. **The auxiliary gradient vanishes at the primary-task optimum**, so dual-head $=$ single-head with extra parameters.

This is why $\lambda_B = 1$, $\lambda_B = 0.1$, detached, and joint all give the same CRPS to four decimals. It is not a tuning failure. It is the v3.x target pair violating a necessary condition.

---

## The necessary condition for a dual-head to add value

$$
\psi^\star_A \;\neq\; \psi^\star_B .
$$

The optimal encoder for the auxiliary target must differ from the optimal encoder for the primary target. One of these must hold:

| Condition | What it means | Example |
|---|---|---|
| **Different horizon** | Different information needs | 1-day vs 20-day vol |
| **Different information source** | Head B sees inputs Head A does not | Cross-asset, options-implied, order flow |
| **Different functional** | Nonlinear functional of the same $X$ | Jumps vs continuous vol, quantile vs mean |
| **Different loss class** | Proper scoring rule the primary cannot mimic | Quantile loss, distributional loss |

The v3.x pair $(\ell^{\mathrm{tot}}, \ell^{\mathrm{down}})$ failed all four: it was the same functional of the same returns over the same horizon, trained under the same loss class. Symmetry identity made it worse — the two targets were related by a fixed offset $c_\star$, so Head B's information was strictly less than Head A's.

---

## Architecture

Single-head is the shipped model. Dual-head is the same trunk with a second readout, used only for hypotheses that pass the $\psi^\star_A \neq \psi^\star_B$ test.

```
X[a,t] ∈ R^(T×12)
      │
      ▼
  LSTM encoder  ψ_θ  ──────┬──────────────┐
                           │              │
                           ▼              ▼
                     Head A  f_A     Head B  f_B
                  (μ_vol, log σ_vol)   (auxiliary target)
                    Gaussian on         chosen to satisfy
                    ℓ^total             ψ*_A ≠ ψ*_B
                           │              │
              L_A (CRPS)  λ_B·L_B  λ_C·L_C (optional)
                           │              │
                           └──► L_total ◄──┘
```

**Head A** — Gaussian on standardized log total volatility, trained under the closed-form **Gaussian CRPS**, a strictly proper scoring rule (Lemma 9.3 in `MATH.md`).

**Head B** — target chosen from the tiered research program below. **Absent by default.**

---

## Cross-cutting improvements

These are worth doing regardless of which auxiliary target is chosen. They are pure improvements — apply them to the single-head first.

| # | Change | Expected CRPS gain | Effort |
|---|---|---|---|
| 1 | **$\sigma$-calibration temperature** — fit a scalar $t^\star$ on validation minimizing coverage error at 80% and 95%; apply $\hat\sigma \mapsto t^\star \hat\sigma$ at test | +2% | 10 min |
| 2 | **EMA of weights** — exponential moving average with $\beta = 0.999$ | +1–2% | 15 min |
| 3 | **Feature engineering** — overnight gap, intraday range, volume imbalance, VIX term spread | +3–5% | 30 min |
| 4 | **Attention pooling** — replace `out[:, -1, :]` with learned attention over timesteps, separate pool per head | +1–3% | 20 min |

**$\sigma$-calibration is the single highest-return-per-minute change.** Current v3.x reliability is $\mathrm{cov}_{80} = 0.780$, $\mathrm{cov}_{95} = 0.926$ — slightly narrow. Under a proper scoring rule, correct calibration is always rewarded.

---

## The tiered research program

Each proposal is a clean falsifiable test. Report against the single-head, HAR-RV, and constant baselines.

### Tier 1 — testable today

**Proposal 1 — Multi-horizon auxiliary.** Head A: $\ell^{\mathrm{tot}}$ at $H=20$. Head B: $\ell^{\mathrm{tot}}$ at $H=1$. Short-horizon vol is dominated by the last 1–5 days of realized variance; long-horizon vol requires regime detection over 20–60 days. The optimal linear projections of $X$ onto each are nearly orthogonal.

*Prediction:* joint beats single by **3–8% CRPS**. Highest-probability win.

**Proposal 2 — Jump–continuous decomposition.** Head A: realized variance $\mathrm{RV} = \sum_h r_h^2$. Head B: bipower variation $\mathrm{BPV} = \tfrac{\pi}{2}\sum_h \lvert r_h\rvert\lvert r_{h-1}\rvert$ (Barndorff-Nielsen & Shephard 2004). The difference $\mathrm{RV} - \mathrm{BPV}$ is the jump component. A window with one large jump has high RV but low BPV; a steady window has $\mathrm{RV} \approx \mathrm{BPV}$.

*Prediction:* joint beats single by **2–5% CRPS** on jump-heavy assets (individual stocks); neutral on broad ETFs.

**Proposal 3 — Cross-sectional rank.** Head A: absolute $\ell^{\mathrm{tot}}$. Head B: rank of $\ell^{\mathrm{tot}}$ within the same calendar date's cross-section. Absolute level is regime-driven (all assets move together in stress); the idiosyncratic component requires encoding relative position.

*Engineering caveat:* batches must contain same-date windows, so the DataLoader needs a `SameDateBatchSampler` instead of random shuffling.

*Prediction:* largest potential gain (**10%+ CRPS**) and highest failure probability.

### Tier 2 — structural

**Proposal 4 — Attention pooling.** Two separate attention pools, one per head, giving each head a different readout of the same LSTM output. Breaks the single-readout degeneracy without changing the encoder. *Prediction:* +1–3% CRPS, independent of auxiliary target.

**Proposal 5 — Heteroscedastic auxiliary (vol-of-vol).** Head B predicts the conditional std of $\ell^{\mathrm{tot}}$ across sub-windows. Requires $H \ge 15$ to have enough sub-samples (e.g. three non-overlapping 5-day blocks). *Prediction:* +2–4% CRPS, complementary with multi-horizon.

### Tier 3 — research

**Proposal 6 — Learned uncertainty weighting (Kendall et al. 2018).** Replace fixed $\lambda_B$ with learned log-variances:

$$
\mathcal{L} = \tfrac{1}{2\sigma_A^2}\mathcal{L}_A + \tfrac{1}{2\sigma_B^2}\mathcal{L}_B + \log\sigma_A + \log\sigma_B .
$$

*Prediction:* stability improvement, neutral in CRPS.

**Proposal 7 — Contrastive auxiliary.** Head B maximizes agreement between differently-augmented views of the same window. *Prediction:* high variance; requires substantial tuning.

---

## Recommended sequence

Run in this order. Each step is a clean falsifiable test.

| Step | Change | Expected CRPS gain | Effort |
|---|---|---|---|
| 1 | $\sigma$-calibration temperature | +2% | 10 min |
| 2 | EMA of weights | +1–2% | 15 min |
| 3 | Feature engineering | +3–5% | 30 min |
| 4 | Attention pooling (both heads) | +1–3% | 20 min |
| 5 | Multi-horizon auxiliary ($H=1$, $H=20$) | +3–8% | 30 min |
| 6 | Jump–continuous decomposition | +2–5% | 30 min |
| 7 | Cross-sectional rank (same-date batching) | +5–10% | 2 hr |

Steps 1–4 are pure improvements; do them regardless. **Step 5 is the first real test of the dual-head hypothesis.** Steps 6–7 are higher-risk.

**Stop condition.** If step 5 does not produce at least a **2% CRPS improvement over single-head**, the dual-head architecture is definitively dead on this problem class. Ship the single-head with steps 1–4 applied and move on.

---

## What not to try

- **More LSTM layers.** 26K params for 64K samples. The model is signal-limited, not depth-limited.
- **Larger hidden size.** You are at the plateau. $h = 128$ will overfit.
- **MoE / more experts.** Same bottleneck — signal, not capacity.
- **Different learning rates.** LR $5\times 10^{-4}$ with early stop at epoch 22 is well-tuned.
- **Ensembling.** Averages out the signal being measured.
- **Isotonic calibration.** The Gaussian CRPS output is already a distribution; isotonic would destroy the vol-of-vol information in $\hat\sigma$.
- **Longer windows ($T = 60, 90$).** Diminishing returns; the recent 20–30 days carry almost all the signal.
- **Direction prediction.** Confirmed dead in v2.0. Features do not carry daily directional signal.

---

## Usage

```bash
pip install -e .
```

```python
from tlstm import SingleHeadTLSTM, Config

cfg = Config(
    window=30,
    horizon=20,
    hidden=64,
    weight_decay=1e-5,
    sigma_temp=True,       # cross-cutting improvement #1
    ema_decay=0.999,       # cross-cutting improvement #2
    attention_pool=True,   # cross-cutting improvement #4
)

model = SingleHeadTLSTM(cfg)
model.fit(panel, split=("2015-01-01", "2021-01-01", "2023-01-01"))
out = model.predict(panel_test)
# out.mu_vol, out.sigma_vol     -> log total vol (standardized)
# out.interval(alpha=0.20)      -> 80% two-sided Gaussian interval
```

To run a dual-head experiment, pass an auxiliary target that satisfies $\psi^\star_A \neq \psi^\star_B$:

```python
from tlstm import DualHeadTLSTM

cfg.head_b = "multi_horizon"   # H=1 auxiliary
# cfg.head_b = "bipower"       # jump-continuous decomposition
# cfg.head_b = "cross_rank"    # requires same-date batching
```

---

## Evaluation

Reported on the test block only. Primary metric is **CRPS** — the only metric in the table that is strictly proper for a distributional forecast.

| Metric | Why it matters |
|---|---|
| CRPS | Proper; rewards location and scale jointly |
| QLIKE | Vol-space Bregman divergence; robust to under-prediction |
| MSE on log vol | Point-forecast comparison |
| Pearson $\rho(\exp(\mu), v^{\mathrm{tot}})$ | Original-unit fit |
| Spearman $\rho_s(\mu, \ell^{\mathrm{tot}})$ | Rank-robust |
| Coverage 80/95 | Reliability curve, not a single number |

**Baselines required** alongside any result:

1. **Constant** — training-block mean of $\ell^{\mathrm{tot}}$.
2. **HAR-RV (Corsi 2009)** — the industry benchmark. The single-head already beats it by **7.5% CRPS** and **17% QLIKE**.
3. **Single-head** ($\lambda_B = 0$) — the model to beat.
4. **Detached Head B** (`sg[ψ]` on encoder path) — isolates co-adaptation from head count.

A claim is reportable only with chronological embargoed splits, train-fold-only preprocessing, a paired binomial or block-bootstrap confidence interval, and all four baselines under the same split.

---

## Status

**Single-head is the working model.** Beats HAR-RV by 7.5% CRPS / 17% QLIKE on the test block. Cross-cutting improvements 1–4 pending.

**Dual-head is a research program.** Stop condition: 2% CRPS improvement over single-head at step 5 (multi-horizon). If not met, ship the single-head and stop.

## License

MIT