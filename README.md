# Dual-Head TLSTM

A shared-encoder, two-head recurrent model for multi-horizon distributional forecasting of log returns on a cross-section of assets — with an auxiliary volatility head and a volatility-thresholded abstention rule.

Every input is $\mathcal{F}_t$-measurable, every split is chronological **by calendar date** and embargoed, and each structural claim in the design carries a proof. See [`MATH.md`](MATH.md).

---

## Architecture

```
X[a,t] ∈ R^(T×12)
      │
      ▼
  LSTM encoder  ψ_θ  ──────┬──────────────┐
                           │              │
                           ▼              ▼
                     Head A  f_A     Head B  f_B
                  (μ_1..μ_H, logσ)    (log v̂)
                           │              │
                    L_A (CRPS)   λ·L_B (vol MSE)
                           │              │
                           └──► L_A + λ_B·L_B + λ_C·L_C ◄──┘
                                   │
                          validation-quantile threshold τ_vol
                                   │
                          { −1, +1, ⊥ }
```

One encoder, two objectives. Gradients from both heads reach $\theta$, so the representation is a compromise between forecasting location and forecasting scale:

$$
\frac{\partial\mathcal{L}_{\mathrm{total}}}{\partial\theta}
= \frac{\partial\mathcal{L}_A}{\partial\psi}\frac{\partial\psi}{\partial\theta}
+ \lambda_B\,\frac{\partial\mathcal{L}_B}{\partial\psi}\frac{\partial\psi}{\partial\theta}
+ \lambda_C\,\frac{\partial\mathcal{L}_C}{\partial\psi}\frac{\partial\psi}{\partial\theta}
$$

---

## Heads

**Head A** — $H$-day Gaussian forecast: location $\mu_h$ and scale $\log\sigma_h$ per horizon, trained under the closed-form **Gaussian CRPS**, a strictly proper scoring rule. Properness is what makes the calibration claim rigorous rather than post hoc.

**Head B** — auxiliary scalar $\log \widehat v$, trained in log space by MSE against realized vol $\sqrt{H^{-1}\sum_h r_{t+h}^2}$. Two roles:

| Role | Output | Feeds abstention |
|---|---|---|
| `vol` (default) | $\widehat v = e^{\widehat \ell}$ | yes — threshold on $\widehat v$ |
| `ablate` | same, but `sg[ψ]` | no — isolates co-adaptation from mere presence of a second head |

An optional consistency term $\mathcal{L}_C$ anchors $\log\bar\sigma$ to `sg[log v̂]`, breaking the additive-shift degeneracy of Head B under $\lambda_C = 0$.

---

## Features

12 causal covariates per asset-day:

$r^{(1)}$, $r^{(5)}$, $r^{(20)}$, $\mathrm{vol}_{20}$, $\mathrm{vol}_{60}$, MA5-ratio, MA20-ratio, RSI(14), relative volume, VIX level, and the percent change of VIX and TNX. Macro series are forward-filled by **release date** only.

---

## Causality guarantees

- Rolling statistics and forward-fill are $\mathcal{F}_t$-measurable; interpolation and backward-fill are not (Prop. 2.7).
- Standardization, vol scale $s_{\mathrm{vol}}$, and the abstention threshold $\tau_{\mathrm{vol}}$ are fitted on the training/validation blocks alone (Lemmas 4.2, 4.5).
- Train/val/test are **contiguous by calendar date**, separated by an embargo of $T + H$ dates, so no window's label crosses a boundary (Thm. 2.10).
- Windows are cut per asset, never over a concatenated panel (Prop. 2.13) — this closes the pooled-index leakage channel left open in v1.0.
- The online replay buffer runs to $s = t - H$, so no label used for an update postdates the forecast (Lemma 13.2).

---

## Usage

```bash
pip install -e .
```

```python
from tlstm import DualHeadTLSTM, Config

cfg = Config(
    window=30,
    horizon=5,
    hidden=64,
    head_b="vol",          # vol | ablate
    lam_B=1.0,             # volatility loss weight
    lam_C=0.0,             # consistency loss weight
    coverage=0.80,         # target κ_0 for abstention
    embargo=True,
)

model = DualHeadTLSTM(cfg)
model.fit(panel, split=("2015-01-01", "2021-01-01", "2023-01-01"))
model.threshold()                 # τ_vol from validation quantile
out = model.predict(panel_test)   # per-horizon (μ, σ), vol, and {−1, +1, ⊥}
```

---

## Key hyperparameters

| Name | Symbol | Role |
|---|---|---|
| `window` | $T$ | sequence length |
| `horizon` | $H$ | forecast horizon in days |
| `hidden` | $h$ | encoder width |
| `lam_B` | $\lambda_B$ | volatility loss weight |
| `lam_C` | $\lambda_C$ | consistency loss weight |
| `coverage` | $\kappa_0$ | target coverage of the abstention rule |
| `weight_decay` | $\gamma$ | AdamW decoupled decay |
| `clip` | $c_\sigma$ | log-$\sigma$ clamp constant |

Notation is deliberately overloaded-free: $\lambda_B, \lambda_C$ are loss weights; $\gamma$ is weight decay; $\mathrm{sigm}$ is the logistic map; $\sigma$ with a subscript is always volatility.

---

## Evaluation

Reported on the test block only: multi-horizon CRPS, directional accuracy, volatility correlation $\mathrm{corr}(\widehat v, y^{\mathrm{vol}})$, **non-overlapping** annualized Sharpe (Def. 15.2 — v1.0's overlapping-window figure overstates by $\sqrt{H}$), coverage $\kappa$, and selective accuracy.

The abstention rule selects a **volatility regime**, not a confidence regime. Coverage is controlled at $\kappa_0$ up to the Kolmogorov distance between the validation and test distributions of $\widehat v$ (Prop. 12.7). Selective accuracy has **no** floor — the bet is that directional accuracy is higher in low-vol regimes, which is a testable hypothesis, not a theorem (Rem. 12.5).

Four baselines are required alongside any result:

1. Majority-direction constant.
2. Single-head ablation ($\lambda_B = \lambda_C = 0$).
3. Head B with `sg[ψ]` on the encoder — isolates co-adaptation (Thm. 10.2) from the mere presence of a second head.
4. No-abstention ablation ($\tau_{\mathrm{vol}} = \infty$).

A claim is reportable only with chronological embargoed splits, train-fold-only preprocessing, a paired binomial or block-bootstrap confidence interval at the realized accepted count, and all four baselines under the same split (Def. 15.6).

---

## Status

Specification complete. Reference implementation in progress. No performance claims are made in this repository until the ablations above are reported.

## License

MIT
