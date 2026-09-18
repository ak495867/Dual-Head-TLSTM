# Dual-Head TLSTM

A shared-encoder recurrent model for **log realized volatility** on a cross-section of assets. Head A emits a flexible conditional distribution; Head B emits an **encoder-separating** auxiliary target. Every input is $\mathcal{F}_t$-measurable, every split is chronological by calendar date and embargoed, and every structural claim carries a proof. See [`MATH.md`](MATH.md).

**Single-head is the shipped model.** Dual-head is a research program with an explicit stop condition.

---

## The structural finding that shapes everything

For any encoder $\psi$ and any auxiliary target $Z_B = g(Z_A, X)$,

$$
I(Z_B; \psi) \;\le\; I(Z_A; \psi) + I(X; \psi).
$$

The primary gradient already maximizes $I(Z_A;\psi)$; the auxiliary gradient adds nothing. **The auxiliary gradient vanishes at the primary-task optimum** (Theorem 10.4), and dual-head collapses to single-head with extra parameters.

This is why v3.x's $(\ell^{\mathrm{tot}}, \ell^{\mathrm{down}})$ pair returned identical CRPS across $\lambda_B \in \{0.1, 1.0\}$, detached, and joint. It is a theorem, not a bug.

**The necessary condition** (Theorem 10.8): a Head-B target can add value only if

$$
\psi^\star_A \;\neq\; \psi^\star_B .
$$

Four sources satisfy this (Theorem 10.12): different **horizon**, different **information source**, different **functional**, different **loss class**.

---

## Architecture

```
X[a,t] ∈ R^(T×d)
      │
      ▼
   LSTM trunk  θ  ──► H_seq ∈ R^(T×h)
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
       attn pool q_A       attn pool q_B        per-head readouts
              │                   │
           adapter / FiLM       adapter / FiLM
              │                   │
              ▼                   ▼
         Head A  f_A          Head B  f_B
    (Gaussian | Student-t |   (multi-horizon |
     skew-t | MoG | quantile)  bipower | rank)
              │                   │
              ▼                   ▼
       L_A (CRPS / pinball)   L_B (matched loss)
              │                   │
              └──► L_A + λ_B L_B + λ_C L_C ◄──┘
```

Head A's working law is a **design variable**, not a fixed Gaussian. Head B's target is chosen from a **restricted menu** of encoder-separating families. Co-adaptation is structured architecturally (per-head attention, adapters, FiLM) and by gradient surgery (PCGrad).

---

## Head A — flexible working laws

The Gaussian is the baseline, not the default. Log realized volatility is left-skewed and bounded below; a symmetric Gaussian cannot represent either feature, and the reliability curve (cov80 = 0.780, cov95 = 0.926 in v3.x) is the signature of the binding constraint.

| Working law | Parameters | CRPS | When to use |
|---|---|---|---|
| Gaussian | 2 | closed form | baseline; short-horizon dense panels |
| Student-$t$ | 3 | closed form | heavier tails |
| Skew-$t$ (Azzalini) | 4 | quadrature | left-skew + tails |
| Mixture of Gaussians ($M=3$) | $3M{-}1$ | closed form | multimodal regimes |
| **Quantile regression** | $K$ | pinball loss | **recommended default** |

**Quantile regression.** Head A emits $K$ monotone quantiles via softplus increments; training minimizes the pinball loss

$$
\mathcal{L}_A = \frac{1}{B}\sum_{(a,t)}\sum_{k=1}^{K} w_k \,\rho_{\tau_k}\big(\widetilde\ell^{\,\mathrm{tot}}_{a,t} - \widehat q_{a,t}(\tau_k)\big),
\qquad \rho_\tau(u) = u(\tau - \mathbf{1}\{u<0\}).
$$

By the CRPS–pinball identity

$$
\mathrm{CRPS}(F,y) = 2\int_0^1 \rho_\tau\big(F^{-1}(\tau),y\big)\,d\tau,
$$

the uniform-grid pinball loss is a Riemann approximation to CRPS over an **unconstrained** distribution family (Theorem 7.12). No Gaussian assumption is imposed. Monotonicity is enforced by the softplus-increment parameterization, which is differentiable and sorting-free.

Default: $K = 19$, grid $\tau_k = k/(K+1)$, $w_k = 1$.

---

## Head B — restricted to separating targets

Three families satisfy Theorem 10.12 and remain informative given the features.

### 1. Multi-horizon term structure (recommended)

$$
\text{Head A: } \ell^{(20)}, \qquad \text{Head B: } (\ell^{(1)}, \ell^{(5)}).
$$

Short-horizon log-vol is activity-driven (last 1–5 days); long-horizon log-vol is regime-driven (20–60 days). The optimal linear projections of $X$ onto each are nearly orthogonal — structural separation by **horizon** (Theorem 10.12(H)). One line of code per asset. **Highest-probability win.**

### 2. Jump–continuous decomposition

$$
\text{Head A: } \ell^{\mathrm{tot}}, \qquad \text{Head B: } \ell^{\mathrm{bpv}}.
$$

Bipower variation (Barndorff-Nielsen & Shephard 2004) differs from realized variance by the jump component $\mathrm{RV} - \mathrm{BPV}$. A window with one large jump has high RV, low BPV; a steady window has RV ≈ BPV. Separation by **functional** (Theorem 10.12(F)). Expect gains on jump-heavy assets (individual stocks), neutrality on broad ETFs.

### 3. Cross-sectional rank

$$
\text{Head A: } \ell^{\mathrm{tot}}, \qquad \text{Head B: } \rho_{a,t} = \frac{\mathrm{rank}_{\mathcal{A}_d}(\ell^{\mathrm{tot}}_{a,t}) - \tfrac12}{\lvert\mathcal{A}_d\rvert}.
$$

Requires a **same-date batch sampler**; random shuffling destroys the cross-section. Separation by **information source** (Theorem 10.12(S)). Highest expected gain (5–10% CRPS), highest failure probability. Audit the sampler before training: every batch same-date, $\lvert\mathcal{A}_d\rvert \ge 8$, target deterministic given the batch.

### What not to use

| Target | Why not |
|---|---|
| $\ell^{\mathrm{down}}$ at the same horizon | $Z_B = Z_A - \tfrac12\log 2$ under symmetry; Theorem 10.4 kills the auxiliary gradient |
| Vol-of-vol across sub-windows | Formally separating but informatively empty; the encoder already represents second-moment information for Head A's scale parameter |

---

## Co-adaptation — structural, not a scalar $\lambda_B$

Three architectural changes with negligible parameter cost, plus one gradient-surgery rule.

**Per-head attention pooling** (Definition 6.4). Two queries $q_A, q_B$ read the same LSTM output tensor with different attention distributions. The last hidden state $h_T$ forces both heads to compromise on what the summary contains; two pools let each attend to the timesteps that matter for its own target.

**Per-head adapters** (Definition 6.5). A residual low-rank MLP between the shared representation and each head:

$$
z^A = \psi^A + \mathrm{Adapter}_A(\psi^A), \qquad \mathrm{Adapter}(\psi) = W_2\,\mathrm{GELU}(W_1\psi).
$$

Cost: $\sim h^2$ parameters per head — about one extra hidden layer.

**FiLM modulation** (Definition 6.7). Diagonal affine map, $4h$ parameters:

$$
z^A = \gamma_A \odot \psi^A + \beta_A .
$$

More expressive than additive adapters for scale-sensitive targets; nearly free.

**PCGrad** (Definition 11.7). When the two heads' encoder gradients conflict — $\langle g_A, g_B\rangle < 0$ — project one onto the normal of the other:

$$
\tilde g_A = g_A - \frac{\langle g_A, g_B\rangle}{\lVert g_B\rVert^2} g_B .
$$

When the gradients agree, PCGrad is a no-op (Proposition 11.8). Measure the conflict frequency first: below 5%, skip it; above 30%, it helps materially.

**Cascade** (Definition 6.9). Head B conditions on `sg[μ_vol]` — use with jump decomposition or cross-sectional rank, not with multi-horizon (levels too correlated).

---

## Features

Reference ($d=12$): $r^{(1)}, r^{(5)}, r^{(20)}, \mathrm{vol}_{20}, \mathrm{vol}_{60}$, MA5-ratio, MA20-ratio, RSI(14), relative volume, VIX level, $\Delta\mathrm{VIX}^{\%}$, $\Delta\mathrm{TNX}^{\%}$.

Extended ($d=16$): add overnight gap $\log(O_t/C_{t-1})$, intraday range $\log(H_t/L_t)$, volume imbalance $(V - \mathrm{med}_{20}(V))/\mathrm{MAD}_{20}(V)$, term spread $\mathrm{VIX} - \mathrm{VIX3M}$.

Macro series are forward-filled by **release date** only (Proposition 2.6).

---

## Causality guarantees

- Rolling statistics and forward-fill are $\mathcal{F}_t$-measurable; interpolation and backward-fill are not (Proposition 2.6).
- Standardization and log-target scaling are fitted on the training block alone (Lemmas 4.2, 4.4).
- Train/val/test are **contiguous by calendar date**, separated by an embargo of $T+H$ dates (Theorem 2.9).
- Windows are cut per asset, never over a concatenated panel (Proposition 2.12).
- Cross-sectional Head-B targets require a same-date batch sampler (Definition 5.4); the encoder itself satisfies $F(\mathcal{X})_b = F_1(\mathcal{X}_b)$ (Constraint 5.3).

---

## Usage

```bash
pip install -e .
```

```python
from tlstm import DualHeadTLSTM, Config

cfg = Config(
    window=30,
    horizon=20,
    hidden=64,
    head_a="quantile",       # gaussian | student_t | skew_t | mog | quantile
    n_quantiles=19,
    head_b="multi_horizon",  # multi_horizon | bipower | cross_rank | none
    pooling="per_head_attn", # last | attn | per_head_attn
    adapters=True,
    pcgrad=True,
    ema_decay=0.999,
    warm_restart_T0=20,
    lam_B=1.0,
    lam_C=0.0,               # only with a functional-coupling target
)

model = DualHeadTLSTM(cfg)
model.fit(panel, split=("2015-01-01", "2021-01-01", "2023-01-01"))
out = model.predict(panel_test)
# out.q(tau)              -> quantile of log total vol at level tau
# out.interval(alpha=0.20) -> 80% two-sided interval
# out.mu_vol, out.sigma_vol -> moment functionals (quadrature)
# out.ell_down, out.rho    -> Head-B outputs when present
```

---

## Key hyperparameters

| Name | Symbol | Role |
|---|---|---|
| `window` | $T$ | sequence length |
| `horizon` | $H$ | forecast horizon in days |
| `hidden` | $h$ | LSTM width |
| `head_a` | — | working-law family |
| `n_quantiles` | $K$ | quantile grid size |
| `head_b` | — | auxiliary target family |
| `pooling` | — | `last`, `attn`, or `per_head_attn` |
| `adapters` | — | per-head low-rank MLP |
| `pcgrad` | — | gradient-conflict projection |
| `lam_B` | $\lambda_B$ | auxiliary loss weight |
| `lam_C` | $\lambda_C$ | consistency offset weight (functional-coupling targets only) |
| `weight_decay` | $\gamma$ | AdamW decoupled decay |
| `clip` | $c_\sigma$ | log-$\sigma$ clamp constant |

Notation is overloaded-free: $\lambda_B,\lambda_C$ are loss weights; $\gamma$ is weight decay; $\mathrm{sigm}$ is the logistic map; $\sigma$ with a subscript is always volatility.

---

## Evaluation

Primary metric is **CRPS** — the only metric in the table strictly proper for a distributional forecast. For the quantile head, the pinball average is CRPS-equivalent under Theorem 7.12.

| Metric | Notes |
|---|---|
| CRPS | proper; primary |
| Pinball average | CRPS-equivalent for quantile head |
| MSE on log vol | point-forecast comparison |
| QLIKE | vol-space Bregman; robust to under-prediction |
| Pearson $\rho(\exp(\mu), v^{\mathrm{tot}})$ | original-unit fit |
| Spearman $\rho_s(\mu, \ell^{\mathrm{tot}})$ | rank-robust |
| Coverage 80 / 95 | report as reliability curve, not a single number |

**Coverage.** For the quantile head, $\lvert\kappa(\alpha) - (1-\alpha)\rvert \le \varepsilon_{\mathrm{q}}$, the worst-case quantile calibration error (Proposition 12.4). Report the quantile reliability diagram $\tau \mapsto \widehat{\mathbb{P}}(\widetilde\ell^{\,\mathrm{tot}} \le \widehat q(\tau))$; under correct quantiles this is the identity.

**Baselines required** alongside any result:

1. **Constant** — training-block empirical distribution of $\widetilde\ell^{\,\mathrm{tot}}$.
2. **HAR-RV (Corsi 2009)** — the industry benchmark. The v3.x single-head already beats it by 7.5% CRPS / 17% QLIKE.
3. **Single-head** ($\lambda_B = \lambda_C = 0$) — the model to beat.
4. **Detached Head B** (`sg[ψ]` on encoder path) — isolates co-adaptation from mere presence of a second head.

A claim is reportable only with chronological embargoed splits, train-fold-only preprocessing, a paired binomial or block-bootstrap confidence interval, and all four baselines under the same split (Definition 15.6). Condition 6: the auxiliary target must be demonstrated encoder-separating, either by construction (Theorem 10.12 case) or by an empirical co-adaptation ablation.

---

## Recommended sequence

| Step | Change | Expected CRPS gain | Effort | Type |
|---|---|---|---|---|
| 1 | Quantile head A ($K=19$) | **5–15%** | 2 hr | Single-head |
| 2 | Per-head attention pool | +1–3% | 30 min | Both |
| 3 | FiLM modulation | +1–2% | 30 min | Both |
| 4 | Loss normalization | +0–1% | 20 min | Both |
| 5 | **Multi-horizon Head B $(20,1)$** | +3–8% | 1 hr | **Dual-head test** |
| 6 | Adapter layers | +1–3% | 1 hr | Requires ≥2 heads |
| 7 | PCGrad | +1–3% if conflict | 1 hr | Measure first |
| 8 | Bipower Head B | +2–5% on jump-heavy assets | 2 hr | Dual-head |
| 9 | Cross-sectional rank Head B | +5–10% if sampler works | 4 hr | Dual-head |

Steps 1–4 are pure single-head improvements. Do them regardless.

**Step 5 is the first real dual-head test.** If quantile-head A plus multi-horizon Head B does not clear **2% CRPS** over single-head with the same Head A, the dual-head hypothesis is rejected for this problem class. Ship the single-head with steps 1–4 applied and stop.

Steps 6–9 only pay off after step 5 succeeds.

---

## What not to try

- **More LSTM layers.** 26K params for 64K samples. Signal-limited, not depth-limited.
- **Larger hidden size.** At the plateau; $h=128$ overfits.
- **MoE / more experts.** Same bottleneck.
- **Different learning rates.** LR $5\times 10^{-4}$ with early stop at epoch 22 is well-tuned.
- **Ensembling.** Averages out the signal being measured.
- **Direction prediction.** Confirmed dead in v2.0.
- **Downside semivolatility as Head B.** Fails Theorem 10.4 (Corollary 10.5); formally dead.
- **Isotonic calibration.** The quantile head is already a distribution; isotonic would destroy the tail information.

---

## Status

**Single-head is the shipped model.** Quantile-head A with $K=19$, pinball loss, per-head attention pooling, EMA, warm restarts. Beats HAR-RV by 7.5% CRPS / 17% QLIKE on the v3.x test block.

**Dual-head is a research program.** Governed by Theorem 10.8 (encoder separation) and Theorem 10.12 (four-source classification). Head B restricted to multi-horizon, bipower, or cross-sectional rank. Stop condition: 2% CRPS over single-head at step 5 (multi-horizon). If unmet, ship the single-head.

## License

MIT
