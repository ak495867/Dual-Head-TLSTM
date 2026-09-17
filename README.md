# Dual-Head TLSTM for Volatility

> Research Question : Does a shared LSTM encoder trained with a downside semivol auxiliary head produce a materially better H day volatility forecast than a 3 parameter HAR-RV regression?
---
A shared-encoder, two-head recurrent model for **log realized volatility** and **log downside semivolatility** on a cross-section of assets — with proper scoring and interval calibration.

Every input is $\mathcal{F}_t$-measurable, every split is chronological **by calendar date** and embargoed, and each structural claim in the design carries a proof. See [`MATH.md`](MATH.md).

---

## What changed and why

The v2.0 run was unambiguous:

| Signal | Evidence | Conclusion |
|---|---|---|
| Vol head works | $\rho = 0.6531$ | Promote to primary |
| Direction head dead | $0.5165 \approx$ constant | Remove entirely |
| Co-adaptation real | Joint $0.6531$ vs Detached $0.6408$ | Keep dual-head structure |
| Encoder encodes vol | $\rho = 0.44$ from random Head B | Encoder genuinely learns vol structure |

The pivot is clean, not a consolation prize. Volatility forecasting is where daily-frequency LSTM models consistently beat HAR-RV baselines; the task has genuine signal-to-noise.

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
                  (μ_vol, log σ_vol)   (log v̂_down)
                    Gaussian on         MSE on
                    log v^total         log v^down
                           │              │
              L_A (CRPS)  λ_B·L_B (MSE)  λ_C·L_C (offset)
                           │              │
                           └──► L_A + λ_B·L_B + λ_C·L_C ◄──┘
```

One encoder, two objectives. Gradients from both heads reach $\theta$, so the representation is a compromise between total-vol and downside-vol structure:

$$
\frac{\partial\mathcal{L}_{\mathrm{total}}}{\partial\theta}
= \frac{\partial\mathcal{L}_A}{\partial\psi}\frac{\partial\psi}{\partial\theta}
+ \lambda_B\,\frac{\partial\mathcal{L}_B}{\partial\psi}\frac{\partial\psi}{\partial\theta}
+ \lambda_C\,\frac{\partial\mathcal{L}_C}{\partial\psi}\frac{\partial\psi}{\partial\theta}
$$

---

## Heads

**Head A** — Gaussian on standardized log **total** volatility, trained under the closed-form **Gaussian CRPS**, a strictly proper scoring rule. Properness is what makes the calibration claim rigorous rather than post hoc.

**Head B** — a **different functional of the same return vector**: log **downside** semivolatility. Trained in log space by MSE. A model that predicts both $v^{\mathrm{tot}}$ and $v^{\mathrm{down}}$ has represented the asymmetry of the return distribution *without ever emitting a directional forecast* — which is exactly the residual signal the v2.0 direction head failed to extract.

**Consistency term** — the two targets are linked by the symmetry identity

$$
\mathbb{E}[r^{2}] \approx 2\,\mathbb{E}[\min(r,0)^{2}]
\;\Longrightarrow\;
\ell^{\mathrm{tot}} \approx \ell^{\mathrm{down}} + \tfrac{1}{2}\log 2 .
$$

The offset $c_\star = \tfrac{1}{2}\log 2 / s_\ell$ is **not a hyperparameter** — it is the population anchor under return symmetry. $\mathcal{L}_C$ penalizes deviation of $\mu^{\mathrm{vol}}$ from `sg[ℓ̂_down + c⋆]`; the stop-gradient is what preserves identifiability.

Residual of $\mathcal{L}_C$ is a leverage-asymmetry diagnostic, reportable without retraining.

---

## Features

12 causal covariates per asset-day:

$r^{(1)}$, $r^{(5)}$, $r^{(20)}$, $\mathrm{vol}_{20}$, $\mathrm{vol}_{60}$, MA5-ratio, MA20-ratio, RSI(14), relative volume, VIX level, and the percent change of VIX and TNX. Macro series are forward-filled by **release date** only.

---

## Causality guarantees

- Rolling statistics and forward-fill are $\mathcal{F}_t$-measurable; interpolation and backward-fill are not (Prop. 2.6).
- Standardization, log-target scaling $(\mu_\ell, s_\ell)$, and the offset $c_\star$ are fitted on the training block alone (Lemmas 4.2, 4.4).
- Train/val/test are **contiguous by calendar date**, separated by an embargo of $T + H$ dates, so no window's label crosses a boundary (Thm. 2.9).
- Windows are cut per asset, never over a concatenated panel (Prop. 2.12) — closes the pooled-index leakage channel left open in v1.0.
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
    lam_B=1.0,            # downside MSE weight
    lam_C=0.5,            # offset consistency weight
    weight_decay=1e-5,    # AdamW decoupled decay γ
)

model = DualHeadTLSTM(cfg)
model.fit(panel, split=("2015-01-01", "2021-01-01", "2023-01-01"))
out = model.predict(panel_test)
# out.mu_vol, out.sigma_vol     -> per-horizon log total vol, standardized
# out.ell_down                  -> log downside semivolatility
# out.interval(alpha=0.20)      -> 80% two-sided Gaussian interval
```

---

## Key hyperparameters

| Name | Symbol | Role |
|---|---|---|
| `window` | $T$ | sequence length |
| `horizon` | $H$ | forecast horizon in days |
| `hidden` | $h$ | encoder width |
| `lam_B` | $\lambda_B$ | downside MSE weight |
| `lam_C` | $\lambda_C$ | offset consistency weight |
| `weight_decay` | $\gamma$ | AdamW decoupled decay |
| `clip` | $c_\sigma$ | log-$\sigma$ clamp constant |

Notation is overloaded-free: $\lambda_B, \lambda_C$ are loss weights; $\gamma$ is weight decay; $\mathrm{sigm}$ is the logistic map; $\sigma$ with a subscript is always volatility.

---

## Evaluation

Reported on the test block only. Primary metric is **CRPS** — the only metric in the table that is strictly proper for a distributional forecast.

| Metric | Why it matters |
|---|---|
| CRPS | Proper; rewards location and scale jointly |
| QLIKE | Vol-space Bregman divergence; robust to under-prediction |
| MSE on log vol | Point-forecast comparison |
| Pearson $\rho(\exp(\mu^{\mathrm{vol}}), v^{\mathrm{tot}})$ | Original-unit fit |
| Spearman $\rho_s(\mu^{\mathrm{vol}}, \ell^{\mathrm{tot}})$ | Rank-robust |
| Coverage 80/95 | Reliability curve, not a single number |

**Interval calibration.** Under a perfectly specified working distribution, $\kappa(\alpha) = 1-\alpha$ (Prop. 12.3). Under misspecification, $\lvert\kappa(\alpha) - (1-\alpha)\rvert \le \varepsilon_{\mathrm{cal}}$, the total-variation distance between the working and true conditional laws (Prop. 12.4). Report coverage as a **reliability curve** $\alpha \mapsto 1-\kappa(\alpha)$ with binomial confidence bands — never a single number in the small-coverage regime.

**Four baselines required** alongside any result:

1. **Constant** — training-block mean of $\widetilde{\ell}^{\,\mathrm{tot}}$.
2. **HAR-RV (Corsi 2009)** — the industry benchmark. *A model that does not beat it on CRPS is not reportable as a volatility forecaster.*
3. **Single-head ablation** ($\lambda_B = \lambda_C = 0$).
4. **Detached Head B** (`sg[ψ]` on the encoder path) — separates co-adaptation (Thm. 10.2) from mere presence of a second head.

The v2.0 result — joint $\rho = 0.6531$ vs detached $\rho = 0.6408$ — is the empirical evidence that co-adaptation, not head count, is doing the work.

A claim is reportable only with chronological embargoed splits, train-fold-only preprocessing, a paired binomial or block-bootstrap confidence interval, and all four baselines under the same split (Def. 15.10).

---

## Status

Specification complete. Reference implementation in progress. No performance claims are made in this repository until the ablations above are reported.

## License

MIT