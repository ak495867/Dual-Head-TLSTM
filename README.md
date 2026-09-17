# Dual-Head TLSTM

A shared-encoder, two-head recurrent model for next-day directional forecasting on a cross-section of assets — with calibrated probabilities and the option to abstain.

Every input is $\mathcal{F}_t$-measurable, every split is chronological and embargoed, and each structural claim in the design carries a proof. See [`DUAL_HEAD_TLSTM`](rough.md).

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
                    (direction)   (error | vol | confidence)
                           │              │
                           └──► L_A + λ·L_B ◄──┘
                                   │
                          isotonic calibration g*
                                   │
                          threshold τ → {0, 1, ⊥}
```

One encoder, two objectives. Gradients from both heads reach $\theta$, so the representation is a compromise between predicting the target and structuring the residual:

$$
\frac{\partial\mathcal{L}_{\mathrm{total}}}{\partial\theta}
= \frac{\partial\mathcal{L}_A}{\partial\psi}\frac{\partial\psi}{\partial\theta}
+ \lambda\,\frac{\partial\mathcal{L}_B}{\partial\psi}\frac{\partial\psi}{\partial\theta}
$$

---

## Head B roles

Exactly one is selected at config time — they imply different losses and different decision rules.

| Role | Output | Loss | Feeds selective layer |
|---|---|---|---|
| `error` | $\widehat e = \widehat y^A - y$ | MSE on residual | no |
| `vol` | $\widehat\sigma > 0$ (softplus) | Gaussian NLL | no |
| `confidence` | $q \approx \mathbb{P}(y = \widehat y^A \mid \mathcal{F}_t)$ | BCE, stop-gradient target | yes |

---

## Features

12 causal covariates per asset-day:

$r^{(1)}$, $r^{(3)}$, $\sigma_{20}$, MA5-ratio, RSI(14), relative volume, VIX, DXY, TNX, and the percent change of each of the three macro series. Macro series are forward-filled by release date only.

---

## Causality guarantees

- Rolling statistics and forward-fill are $\mathcal{F}_t$-measurable; interpolation and backward-fill are not.
- Standardization is fitted on the training block alone.
- Train/val/test are contiguous and separated by an embargo of $T$ indices, so no window's label crosses a boundary.
- Windows are cut per asset, never over a concatenated panel.
- The online replay buffer runs to $s = t-1$, so no label used for an update postdates the forecast.

---

## Usage

```bash
pip install -e .
```

```python
from tlstm import DualHeadTLSTM, Config

cfg = Config(
    window=30,
    hidden=64,
    head_b="confidence",   # error | vol | confidence
    lam=0.5,               # auxiliary loss weight
    tau=0.60,              # selective threshold
    embargo=True,
)

model = DualHeadTLSTM(cfg)
model.fit(panel, split=("2015-01-01", "2021-01-01", "2023-01-01"))
model.calibrate()          # isotonic, validation block
pred = model.predict(panel_test)   # values in {0, 1, ⊥}
```

---

## Key hyperparameters

| Name | Symbol | Role |
|---|---|---|
| `window` | $T$ | sequence length |
| `hidden` | $h$ | encoder width |
| `lam` | $\lambda$ | auxiliary loss weight |
| `tau` | $\tau$ | abstention threshold, $\tau \in [0.5, 1)$ |
| `weight_decay` | $\gamma$ | AdamW decoupled decay |
| `ewc_rho` | $\rho$ | online stability penalty |

---

## Evaluation

Reported on the test block only: accuracy, AUC, Brier, ECE, coverage $\kappa$, selective accuracy.

Under perfect calibration and $\tau \ge \tfrac12$, selective accuracy is bounded below by $\tau$; with calibration error $\varepsilon_{\mathrm{cal}}$ the bound degrades to $\tau - \varepsilon_{\mathrm{cal}}/\kappa$, so low-coverage regimes need wide confidence intervals rather than point estimates.

Three baselines are required alongside any result: majority-class constant, $\lambda = 0$ single-head, and Head B with $\psi$ detached. The third is what separates co-adaptation from the mere presence of a second head.

---

## Status

Specification complete. Reference implementation in progress. No performance claims are made in this repository until the ablations above are reported.

## License

MIT
