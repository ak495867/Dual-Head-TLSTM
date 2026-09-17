# Dual Head TLSTM: Mathematical Specification

## A co-adaptive, calibrated, selective forecasting engine

---

## 1. Notation and probabilistic setup

Let \((\Omega,\mathcal F,\mathbb P)\) be a probability space equipped with a filtration \(\{\mathcal F_t\}_{t\in\mathbb Z}\), where \(\mathcal F_t\) represents **all information available at the close of trading day \(t\)**. The filtration is increasing:

\[
\mathcal F_s \subseteq \mathcal F_t \qquad \text{for } s \le t.
\]

Consider a universe of assets \(\mathcal A\). For asset \(a\in\mathcal A\), let

\[
C_{a,t}
\]

denote the closing price on day \(t\), and \(V_{a,t}\) the traded volume. Both are \(\mathcal F_t\)-measurable.

Define the **direction label**

\[
y_{a,t+1}
=
\mathbf 1\{C_{a,t+1} > C_{a,t}\}.
\]

Note \(y_{a,t+1}\) is \(\mathcal F_{t+1}\)-measurable but **not** \(\mathcal F_t\)-measurable. This is the fundamental causality constraint.

---

## 2. Causality and the no-leakage principle (formal)

### 2.1 Definitions

**Causal feature map.** A feature map

\[
\varphi_t : \Omega \to \mathbb R^d
\]

is **causal** (or \(\mathcal F_t\)-adapted) if \(\varphi_t\) is \(\mathcal F_t\)-measurable for every \(t\).

**Non-anticipative predictor.** A predictor \(\widehat y_{t+1}\) is **non-anticipative** if it is \(\mathcal F_t\)-measurable.

**Look-ahead leakage.** A model exhibits look-ahead leakage if any of its outputs at time \(t\) depend on \(\mathcal F_{t+k}\) for \(k > 0\).

### 2.2 Lemma (Composition preserves causality)

If \(\varphi_1,\dots,\varphi_n\) are each \(\mathcal F_t\)-measurable and \(g\) is Borel-measurable, then \(g(\varphi_1,\dots,\varphi_n)\) is \(\mathcal F_t\)-measurable.

**Proof.** Each \(\varphi_i\) is measurable w.r.t. \(\mathcal F_t\). Any Borel function of measurable functions is measurable w.r.t. the same \(\sigma\)-algebra. \(\square\)

### 2.3 Corollary (Rolling statistics are causal)

Any rolling statistic computed from \(\{x_{t-k}\}_{k\ge 0}\), such as

\[
\sigma_{20,t}
=
\operatorname{sd}(r_{t-19},\dots,r_t),
\qquad
\operatorname{MA}_{14}(U)_t
=
\frac{1}{14}\sum_{j=0}^{13}U_{t-j},
\]

is \(\mathcal F_t\)-measurable and therefore causal.

### 2.4 Corollary (Forward-fill is causal)

If a macro series \(M_\tau\) is observed at times \(\tau\), then the forward-filled series

\[
\widetilde M_t = M_{\tau(t)},
\qquad
\tau(t)=\max\{\tau\le t\},
\]

is \(\mathcal F_t\)-measurable. Backward-fill or interpolation is not.

### 2.5 Chronological split as a \(\sigma\)-algebra separation

Let the training, validation, and test index sets be

\[
\mathcal T_{\text{tr}} = \{1,\dots,T_1\},
\quad
\mathcal T_{\text{va}} = \{T_1+1,\dots,T_2\},
\quad
\mathcal T_{\text{te}} = \{T_2+1,\dots,T_3\}.
\]

**No-leakage condition.** For every parameter \(\theta\) estimated using \(\mathcal T_{\text{tr}}\),

\[
\theta \text{ is } \mathcal F_{T_1}\text{-measurable}.
\]

For every calibration map \(g\) estimated using \(\mathcal T_{\text{va}}\),

\[
g \text{ is } \mathcal F_{T_2}\text{-measurable}.
\]

Evaluation on \(\mathcal T_{\text{te}}\) uses only these measurable objects. Therefore no test-time information influences training. \(\square\)

### 2.6 Corollary (Per-ticker windowing)

Windows must be constructed per ticker:

\[
X_{a,t}
=
[\varphi_{t-T+1},\dots,\varphi_t]\in\mathbb R^{T\times d}.
\]

Concatenating features across tickers before windowing creates boundaries where a window crosses two assets. Such a window is not \(\mathcal F_t\)-measurable w.r.t. a single asset and violates the causality guarantee.

---

## 3. Feature construction

For each asset and day \(t\), construct the causal feature vector

\[
\mathbf x_t=
\begin{bmatrix}
r_t^{(1)} &
r_t^{(3)} &
\sigma_{20,t} &
\operatorname{MA5Ratio}_t &
\operatorname{RSI}_t &
\Delta V_t^{\%} &
\mathrm{VIX}_t &
\mathrm{DXY}_t &
\mathrm{TNX}_t &
\Delta \mathrm{VIX}_t^{\%} &
\Delta \mathrm{DXY}_t^{\%} &
\Delta \mathrm{TNX}_t^{\%}
\end{bmatrix}^\top
\in\mathbb R^{12}.
\]

Every component is \(\mathcal F_t\)-measurable.

### 3.1 Standardization (train-fold only)

Let \(\mathcal T_{\text{tr}}\) be the model-fitting subset. Fit

\[
\mu_j = \frac{1}{|\mathcal T_{\text{tr}}|}\sum_{t\in\mathcal T_{\text{tr}}}x_{t,j},
\qquad
\sigma_j^2 = \frac{1}{|\mathcal T_{\text{tr}}|-1}\sum_{t\in\mathcal T_{\text{tr}}}(x_{t,j}-\mu_j)^2.
\]

Then

\[
\widetilde x_{t,j} = \frac{x_{t,j}-\mu_j}{\sigma_j+\epsilon}
\]

for all \(t\). The transformed features on the validation and test folds are \(\mathcal F_t\)-measurable but use only training-fold statistics for the affine map. \(\square\)

### 3.2 Lemma (Standardization preserves causality)

If \(\mu_j,\sigma_j\) are computed only from \(\mathcal T_{\text{tr}}\), then the affine map \(x\mapsto (x-\mu_j)/(\sigma_j+\epsilon)\) is deterministic and \(\mathcal F_t\)-measurable at evaluation time. \(\square\)

---

## 4. Window construction and tensor algebra

Let \(T\) be the window length and \(d=12\) the feature dimension. For asset \(a\):

\[
X_{a,t}=
[\widetilde{\mathbf x}_{t-T+1},\dots,\widetilde{\mathbf x}_t]
\in\mathbb R^{T\times d}.
\]

The target aligned with \(X_{a,t}\) is \(y_{a,t+1}\).

For a minibatch of \(B\) windows:

\[
\mathcal X \in \mathbb R^{B\times T\times d}.
\]

No transformation is applied across the batch dimension that mixes assets.

---

## 5. Shared LSTM encoder

### 5.1 LSTM cell equations

Let \(h\in\mathbb N\) be the hidden size and \(d=12\). For step \(i\in\{1,\dots,T\}\):

\[
\begin{aligned}
f_i &= \sigma(W_f x_i + U_f h_{i-1} + b_f),\\
i_i &= \sigma(W_i x_i + U_i h_{i-1} + b_i),\\
o_i &= \sigma(W_o x_i + U_o h_{i-1} + b_o),\\
\tilde c_i &= \tanh(W_c x_i + U_c h_{i-1} + b_c),\\
c_i &= f_i\odot c_{i-1} + i_i \odot \tilde c_i,\\
h_i &= o_i \odot \tanh(c_i).
\end{aligned}
\]

All weight matrices are of compatible shape. The **encoder output** is

\[
h_T = \mathrm{LSTM}_\theta(X_{a,t}).
\]

Optionally pool over time:

\[
\bar h = \frac{1}{T}\sum_{i=1}^{T}h_i.
\]

Denote by

\[
\psi_\theta(X_{a,t}) \in \mathbb R^h
\]

the fixed representation passed to both heads (either \(h_T\) or \(\bar h\)).

### 5.2 Lemma (Encoder is non-anticipative)

Since each \(x_i\) for \(i \le T\) is \(\mathcal F_t\)-measurable and the LSTM is a deterministic function of the inputs, \(\psi_\theta(X_{a,t})\) is \(\mathcal F_t\)-measurable. \(\square\)

---

## 6. Head A — primary output

Head A produces the primary prediction. Let \(f_A(\cdot;\theta_A):\mathbb R^h\to\mathbb R\) be a linear (or MLP) map.

**Regression form:**

\[
\widehat y^A_{a,t+1} = f_A(\psi_\theta(X_{a,t});\theta_A).
\]

**Classification form:**

\[
p^A_{a,t+1} = \sigma\!\big(f_A(\psi_\theta(X_{a,t});\theta_A)\big),
\qquad
\widehat y^A_{a,t+1}=\mathbf 1\{p^A_{a,t+1}\ge 0.5\}.
\]

Head A is \(\mathcal F_t\)-measurable.

---

## 7. Head B — auxiliary output

Head B serves one of three interchangeable roles. All three are supported by the same backbone.

### 7.1 Error-correction head (Spec 1 style)

Predict the signed error of Head A:

\[
e_{a,t+1} = \widehat y^A_{a,t+1} - y_{a,t+1},
\qquad
\widehat e_{a,t+1} = f_B(\psi_\theta(X_{a,t});\theta_B).
\]

Corrected forecast:

\[
\widehat y^{\text{final}}_{a,t+1} = \widehat y^A_{a,t+1} - \widehat e_{a,t+1}.
\]

### 7.2 Volatility head (risk-parallel style)

\[
\widehat \sigma_{a,t+1} = \exp\!\big(f_B(\psi_\theta(X_{a,t});\theta_B)\big).
\]

Uses a softplus or exp parameterization to guarantee positivity.

### 7.3 Confidence head (selective-prediction style)

Predict the probability that Head A is correct:

\[
q_{a,t+1} = \sigma\!\big(f_B(\psi_\theta(X_{a,t});\theta_B)\big)
\approx P\!\big(Y_{a,t+1}=\widehat y^A_{a,t+1}\mid \mathcal F_t\big).
\]

### 7.4 Head B is non-anticipative

Same argument as §5.2: \(f_B\) and \(\psi_\theta\) are \(\mathcal F_t\)-measurable. \(\square\)

---

## 8. Composite loss

### 8.1 Error-correction instantiation

\[
\mathcal L_A
=
\frac{1}{B}\sum_{a,t}\big(y_{a,t+1}-\widehat y^A_{a,t+1}\big)^2,
\]

\[
\mathcal L_B
=
\frac{1}{B}\sum_{a,t}\big(e_{a,t+1}-\widehat e_{a,t+1}\big)^2
=
\frac{1}{B}\sum_{a,t}\big(y_{a,t+1}-\widehat y^{\text{final}}_{a,t+1}\big)^2.
\]

Total:

\[
\mathcal L_{\text{total}}=\mathcal L_A + \lambda\,\mathcal L_B,
\qquad \lambda>0.
\]

### 8.2 Classification instantiation

\[
\mathcal L_A = \mathrm{BCE}(y_{a,t+1},p^A_{a,t+1}),
\]

\[
\mathcal L_B = \mathrm{BCE}\big(\mathbf 1\{y_{a,t+1}=\widehat y^A_{a,t+1}\},q_{a,t+1}\big).
\]

### 8.3 Role of \(\lambda\)

The auxiliary term prevents degenerate solutions in which Head A collapses (e.g., outputs zero) while Head B supplies all correction signal.

---

## 9. Gradient flow and co-adaptation

### 9.1 Gradient decomposition

The encoder parameters \(\theta\) appear in \(\mathcal L_A\) and \(\mathcal L_B\) through \(\psi_\theta\). By the chain rule:

\[
\frac{\partial \mathcal L_{\text{total}}}{\partial\theta}
=
\underbrace{
\frac{\partial \mathcal L_A}{\partial \widehat y^A}
\frac{\partial \widehat y^A}{\partial \psi}
\frac{\partial \psi}{\partial\theta}
}_{\text{primary path}}
+
\lambda\,
\underbrace{
\frac{\partial \mathcal L_B}{\partial \widehat e}
\frac{\partial \widehat e}{\partial \psi}
\frac{\partial \psi}{\partial\theta}
}_{\text{auxiliary path}}.
\]

### 9.2 Theorem (Co-adaptation via shared representation)

Let \(\psi_\theta\) be the shared encoder and let \(\mathcal L_A,\mathcal L_B\) be differentiable losses. Then the gradient update

\[
\theta \leftarrow \theta - \eta\,\frac{\partial \mathcal L_{\text{total}}}{\partial\theta}
\]

shapes \(\psi_\theta\) to serve **both** heads simultaneously.

**Proof.** The total derivative decomposes as in §9.1. Both terms involve \(\partial\psi/\partial\theta\), so the encoder receives gradients from both tasks. The direction of steepest descent for \(\mathcal L_{\text{total}}\) is the sum of the two task-specific directions. Hence \(\theta\) is updated to reduce the joint loss, not either loss alone. \(\square\)

### 9.3 Corollary (Error-predictability incentive)

Assume the error-correction instantiation (§7.1) and that Head B is optimal given \(\psi\). Then the achievable auxiliary loss is

\[
\inf_{\theta_B}\mathcal L_B
=
\frac{1}{B}\sum_{a,t}\mathrm{Var}\!\big(e_{a,t+1}\mid \psi_\theta(X_{a,t})\big).
\]

If the encoder changes from \(\psi\) to \(\psi'\), the achievable auxiliary loss decreases iff

\[
\mathbb E\big[\mathrm{Var}(e\mid\psi')\big]
<
\mathbb E\big[\mathrm{Var}(e\mid\psi)\big],
\]

i.e., iff \(\psi'\) captures strictly more information about the predictable component of \(e\). Gradients from \(\mathcal L_B\) therefore push the encoder toward representations that make Head A's errors predictable.

**Proof.** For any measurable \(g\),

\[
\mathbb E\big[(e-g(\psi))^2\big]
=
\mathbb E\big[\mathrm{Var}(e\mid\psi)\big]
+
\mathbb E\big[(\mathbb E[e\mid\psi]-g(\psi))^2\big].
\]

The first term is irreducible given \(\psi\); the second vanishes at \(g^*(\psi)=\mathbb E[e\mid\psi]\). Minimizing over \(\theta_B\) gives the stated infimum. \(\square\)

### 9.4 Formal co-adaptation

Since \(\theta\) receives gradient from \(\mathcal L_B\), which is minimized when \(\psi_\theta\) maximizes predictability of \(e\), the encoder is incentivized to:

1. Reduce Head A's error (via \(\mathcal L_A\)).
2. Structure the remaining error in a way Head B can predict (via \(\mathcal L_B\)).

This is a Nash-like equilibrium between the two heads. \(\square\)

---

## 10. Information-theoretic view

### 10.1 Data processing inequality

Let \(e\) be the primary error, \(X\) the input window, and \(\psi=\psi_\theta(X)\) the encoder output. Since \(\psi\) is a deterministic function of \(X\),

\[
I(e;\psi) \le I(e;X).
\]

Equality holds iff \(\psi\) is a sufficient statistic for \(e\).

### 10.2 Mutual-information maximization interpretation

Training with \(\mathcal L_B\) increases \(I(e;\psi)\) toward its upper bound \(I(e;X)\) by removing nuisance variation in \(\psi\) that does not inform \(e\).

Formally, minimizing \(\mathbb E[\mathrm{Var}(e\mid\psi)]\) is equivalent to maximizing the explained variance

\[
\mathrm{Var}(e) - \mathbb E[\mathrm{Var}(e\mid\psi)]
=
I_{\text{lin}}(e;\psi),
\]

the linear-in-\(L^2\) analogue of mutual information.

---

## 11. Isotonic calibration

### 11.1 Setup

Let \(s_i\) be the score of the meta-learner on validation example \(i\). Isotonic regression estimates a non-decreasing \(g:\mathbb R\to\mathbb R\) minimizing

\[
g^* = \arg\min_{g\text{ non-decreasing}}\sum_{i=1}^{n}(g(s_i)-y_i)^2.
\]

### 11.2 Theorem (PAV optimality)

The pool-adjacent-violators algorithm produces the global minimizer of the isotonic least-squares problem.

**Proof.** After sorting the distinct scores \(s_{(1)}<\cdots<s_{(m)}\), the problem is a convex quadratic program

\[
\min_{u_1\le\cdots\le u_m}\sum_{j=1}^{m}(u_j-\bar y_j)^2.
\]

KKT conditions: for every adjacent pair \((j,j+1)\),

\[
(u_j-\bar y_j)=\mu_j,\quad
(u_{j+1}-\bar y_{j+1})=-\mu_j,\quad
\mu_j(u_{j+1}-u_j)=0.
\]

Thus either \(u_j<u_{j+1}\) with \(u_j=\bar y_j\) and \(u_{j+1}=\bar y_{j+1}\), or \(u_j=u_{j+1}\) and the common value is the pooled mean. PAV merges blocks exactly when the raw means violate monotonicity and assigns each block its label mean, satisfying the KKT conditions. Convexity gives global optimality. \(\square\)

### 11.3 Rank preservation

If \(g^*\) is strictly increasing, AUC is preserved. If \(g^*\) is merely non-decreasing, ties are introduced and AUC can change by at most the mass of merged blocks.

### 11.4 MSE decomposition

For any estimator \(g\),

\[
\mathbb E[(y-g(s))^2]
=
\underbrace{\mathbb E[(y-p(s))^2]}_{\text{irreducible}}
+
\underbrace{\mathbb E[(p(s)-g(s))^2]}_{\text{calibration error}},
\]

where \(p(s)=\mathbb E[y\mid s]\). Isotonic regression minimizes the second term among non-decreasing \(g\).

---

## 12. Selective prediction

### 12.1 Decision rule

Given calibrated \(p_{\text{cal}}\) and threshold \(\tau\in(0.5,1)\):

\[
\widehat y=
\begin{cases}
1, & p_{\text{cal}}\ge\tau,\\
0, & p_{\text{cal}}\le 1-\tau,\\
\bot, & \text{otherwise (abstain)}.
\end{cases}
\]

### 12.2 Coverage

\[
\kappa = \mathbb P(\widehat y\ne\bot).
\]

### 12.3 Selective accuracy

\[
\mathrm{Acc}_{\text{sel}}
=
\mathbb P(\widehat y=Y\mid\widehat y\ne\bot).
\]

### 12.4 Theorem (accuracy floor under perfect calibration)

If \(p(X)=P(Y=1\mid X)\) is perfectly calibrated and \(\tau\ge 0.5\), then

\[
\mathrm{Acc}_{\text{sel}}\ge \tau.
\]

**Proof.** Let \(A\) denote the event that the policy acts. On \(\{p(X)\ge\tau\}\), the prediction is \(1\) and it is correct with probability \(p(X)\ge\tau\). On \(\{p(X)\le 1-\tau\}\), the prediction is \(0\) and it is correct with probability \(1-p(X)\ge\tau\). Therefore

\[
\mathbb E[\mathbf 1\{\widehat y=Y\}\mathbf 1\{A\}]
\ge \tau\,\mathbb P(A).
\]

Dividing by \(\mathbb P(A)\) gives the result. \(\square\)

### 12.5 Corollary (coverage–precision trade-off)

As \(\tau\to 0.5^+\), \(\kappa\to 1\). As \(\tau\to 1^-\), \(\kappa\to 0\). Accuracy of the selected set is non-decreasing in \(\tau\) under perfect calibration.

### 12.6 Finite-sample caveat

Under estimated calibration, the observed \(\mathrm{Acc}_{\text{sel}}\) may fall below \(\tau\) by an amount bounded by the calibration error plus sampling noise. This does **not** violate the theorem; it reflects estimation uncertainty.

---

## 13. Online co-adaptation

### 13.1 Rolling buffer

Maintain

\[
\mathcal B_t = \{(X_{a,s},y_{a,s+1}) : s\in[t-M+1,t]\}.
\]

### 13.2 Online update

After observing \(y_{a,t+1}\), perform \(K\) gradient steps:

\[
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal L_{\text{total}}(\mathcal B_t).
\]

### 13.3 Regularization for stability

- Small learning rate \(\eta_{\text{online}}\ll\eta_{\text{offline}}\).
- Replay buffer of older samples.
- Elastic weight consolidation:

\[
\mathcal L_{\text{EWC}}(\theta)
=
\mathcal L_{\text{total}}(\theta)
+
\frac{\rho}{2}\sum_k F_k(\theta_k-\theta_k^{\text{ref}})^2,
\]

where \(F_k\) is the Fisher information of parameter \(k\) and \(\theta^{\text{ref}}\) is the offline-trained parameter.

### 13.4 Preservation of causality

All updates use only observations in \(\mathcal B_t\), which are \(\mathcal F_t\)-measurable. No future information enters. \(\square\)

---

## 14. Convergence and stability

### 14.1 AdamW update

\[
m_k = \beta_1 m_{k-1} + (1-\beta_1)g_k,
\quad
v_k = \beta_2 v_{k-1} + (1-\beta_2)g_k^2,
\]

\[
\widehat m_k = \frac{m_k}{1-\beta_1^k},
\quad
\widehat v_k = \frac{v_k}{1-\beta_2^k},
\]

\[
\theta_{k+1} = \theta_k - \eta\,\frac{\widehat m_k}{\sqrt{\widehat v_k}+\epsilon} - \eta\lambda\,\theta_k.
\]

The decoupled weight decay acts as a proximal \(L^2\) penalty.

### 14.2 Non-convexity

\(\mathcal L_{\text{total}}\) is non-convex because of the LSTM recurrence. Convergence is to a stationary point:

\[
\|\nabla_\theta \mathcal L_{\text{total}}(\theta^*)\| \le \varepsilon.
\]

No global optimality guarantee exists. This is standard for recurrent neural networks.

### 14.3 Early stopping

Use validation loss with patience \(P\):

\[
k^* = \arg\min_{k\le K}\mathcal L_{\text{val}}(\theta_k).
\]

Stop when validation loss has not improved for \(P\) consecutive epochs. \(\theta_{k^*}\) is the reported model.

---

## 15. Summary of the mathematical engine

| Component | Mathematical role | Output |
|---|---|---|
| Filtration \(\{\mathcal F_t\}\) | Formalizes information at time \(t\) | — |
| Causal feature map \(\varphi_t\) | \(\mathcal F_t\)-measurable covariates | \(\mathbf x_t\in\mathbb R^{12}\) |
| Window \(X_{a,t}\) | Local temporal context | \(\mathbb R^{T\times d}\) |
| Encoder \(\psi_\theta\) | Non-anticipative representation | \(\mathbb R^h\) |
| Head A \(f_A\) | Primary prediction | \(\widehat y^A\) or \(p^A\) |
| Head B \(f_B\) | Auxiliary prediction | \(\widehat e\), \(\widehat\sigma\), or \(q\) |
| Composite loss | Joint training signal | \(\mathcal L_A+\lambda\mathcal L_B\) |
| Gradient flow | Co-adaptation mechanism | \(\partial\psi/\partial\theta\) from both heads |
| MI view | Information-theoretic justification | \(\max I(e;\psi)\) |
| Isotonic calibration | Monotone probability correction | \(p_{\text{cal}}\) |
| Selective rule | Act or abstain | \(\widehat y\in\{0,1,\bot\}\) |
| Online updates | Regime adaptation | \(\theta\leftarrow\theta-\eta\nabla\mathcal L\) |
| AdamW + early stop | Optimization protocol | Converged \(\theta^*\) |

The engine separates four mathematically distinct concerns:

1. **Causality**: every input is \(\mathcal F_t\)-measurable and every split is chronological.
2. **Representation**: a shared LSTM produces a single latent state used by both heads.
3. **Estimation**: each head is a supervised map from the shared state to its target.
4. **Decision**: calibration and thresholding convert probability into action or abstention.

Each stage has an explicit proof obligation causality (2.5), non-anticipativity (5.2), co-adaptation (9.2–9.4), calibration optimality (11.2), and selective accuracy (12.4) and each is satisfied within the stated assumptions.