"""
Dual-Head TLSTM v5.0 — Quantile Head A + Multi-Horizon Head B
=============================================================
Implements the three levers from "Adapting Both Heads":

  Lever A — Head A: quantile regression over ℓ^(H_A=20), K=39 quantiles.
            Pinball loss = Riemann approximation to CRPS.

  Lever B — Head B: multi-horizon Gaussian on ℓ^(H=1) and ℓ^(H=5).
            Satisfies Thm 10.12(H) encoder-separation.

  Lever C — Per-head attention pooling (C.3a)
            FiLM modulation (C.3c)
            Running-mean loss normalization (C.3e)

Causality identical to v4.0:
  §2.4 Calendar-date split + embargo T + H_max
  §2.5 Per-asset windows (no cross-ticker)
  §4.1 Train-fold-only feature standardization
  §4.2 Train-fold-only target standardization

Baselines: HAR-RV (H_A), single-head quantile, detached Head B, constant.

Saves: dual_head_tlstm_v5.pth
"""

import os, math, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==================================================================
# Reproducibility
# ==================================================================
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ==================================================================
# Config
# ==================================================================
TRAIN_TICKERS = [
    'SPY','QQQ','IWM','DIA','MDY','RSP','VTI',
    'XLE','XLF','XLK','XLV','XLI','XLP','XLY','XLU','XLB',
    'AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA','JPM',
    'JNJ','WMT','XOM','BAC','V','UNH','PG','HD','CVX','ABBV',
    'KO','PEP','ORCL','CSCO'
]
START_DATE = '2012-01-01'
END_DATE   = '2023-12-31'

T          = 30
H_A        = 20               # primary (regime-driven)
H_B1       = 1                # auxiliary short (activity-driven)
H_B5       = 5                # auxiliary mid
H_MAX      = max(H_A, H_B5)

HIDDEN     = 64
NUM_LAYERS = 1
DROPOUT    = 0.1

# Quantile grid: uniform from 0.025 to 0.975 in steps of 0.025 → K = 39
TAU_GRID   = np.linspace(0.025, 0.975, 39).astype(np.float32)
K_QUANTILES= len(TAU_GRID)
Q_INIT_BIAS= -2.0              # softplus(-2) ≈ 0.127, sum over 38 increments ≈ 4.8

# Training
BATCH_SIZE = 512
EPOCHS     = 150
LR         = 5e-4
GAMMA      = 1e-5
LAMBDA_B   = 0.5
PATIENCE   = 20
EMA_DECAY  = 0.999
WARM_T0    = 40
LOSS_MOMENTUM = 0.99           # running-mean momentum for loss normalization

EPS        = 1e-6
CKPT_PATH  = 'dual_head_tlstm_v5.pth'
CACHE_DIR  = './cache_v5'
os.makedirs(CACHE_DIR, exist_ok=True)

FEATURE_COLS = [
    'r1','r5','r20','vol20','vol60',
    'ma5_ratio','ma20_ratio','rsi',
    'vol_pct','vol_imbalance',
    'gap','intraday_range',
    'vix','vix_pct','tnx_pct','vix_term',
]
HAR_COLS = ['rv_d','rv_w','rv_m']

# ==================================================================
# Data
# ==================================================================
def download_prices(ticker, start, end):
    path = os.path.join(CACHE_DIR, f"{ticker}_{start}_{end}.parquet")
    if os.path.exists(path):
        try: return pd.read_parquet(path)
        except Exception: pass
    try:
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
    except Exception:
        return None
    if df is None or len(df) < 500: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    needed = ['Open','High','Low','Close','Volume']
    if not all(c in df.columns for c in needed): return None
    df = df[needed].dropna()
    try: df.to_parquet(path)
    except Exception: pass
    return df


def download_macro(start, end):
    path = os.path.join(CACHE_DIR, f"macro_{start}_{end}.parquet")
    if os.path.exists(path):
        try: return pd.read_parquet(path)
        except Exception: pass
    out = {}
    for sym, name in [('^VIX','vix'), ('^VIX3M','vix3m'), ('^TNX','tnx')]:
        try:
            d = yf.download(sym, start=start, end=end,
                            auto_adjust=True, progress=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            out[name] = d['Close']
        except Exception:
            out[name] = pd.Series(dtype=float)
    macro = pd.DataFrame(out)
    try: macro.to_parquet(path)
    except Exception: pass
    return macro


def compute_features(df, macro):
    close  = df['Close'].astype(float)
    open_  = df['Open'].astype(float)
    high   = df['High'].astype(float)
    low    = df['Low'].astype(float)
    volume = df['Volume'].astype(float)
    logc   = np.log(close)
    f = pd.DataFrame(index=df.index)

    f['r1']  = logc.diff(1)
    f['r5']  = logc.diff(5)
    f['r20'] = logc.diff(20)
    f['vol20'] = f['r1'].rolling(20).std()
    f['vol60'] = f['r1'].rolling(60).std()
    f['ma5_ratio']  = close / close.rolling(5).mean()  - 1.0
    f['ma20_ratio'] = close / close.rolling(20).mean() - 1.0

    delta = close.diff()
    U = delta.clip(lower=0); D = (-delta).clip(lower=0)
    n = 14
    Ub = U.ewm(alpha=1.0/n, adjust=False).mean()
    Db = D.ewm(alpha=1.0/n, adjust=False).mean()
    f['rsi'] = 100.0 * (1.0 - 1.0 / (1.0 + Ub / (Db + 1e-9)))

    f['vol_pct'] = volume / (volume.rolling(20).mean() + 1e-9) - 1.0
    vmed = volume.rolling(20).median()
    vmad = (volume - vmed).abs().rolling(20).median() + 1e-6
    f['vol_imbalance'] = (volume - vmed) / vmad

    f['gap']            = np.log(open_ / close.shift(1) + 1e-9)
    f['intraday_range'] = np.log(high / (low + 1e-9))

    macro_ff = macro.reindex(f.index, method='ffill')
    f['vix']     = macro_ff['vix']
    f['vix_pct'] = macro_ff['vix'].pct_change(1)
    f['tnx_pct'] = macro_ff['tnx'].pct_change(1)
    if 'vix3m' in macro_ff.columns and macro_ff['vix3m'].notna().sum() > 100:
        f['vix_term'] = macro_ff['vix'] - macro_ff['vix3m']
    else:
        f['vix_term'] = 0.0

    r2 = f['r1'] ** 2
    f['rv_d'] = r2
    f['rv_w'] = r2.rolling(5).mean()
    f['rv_m'] = r2.rolling(22).mean()

    f['log_ret'] = f['r1']
    f = f.replace([np.inf, -np.inf], np.nan).dropna()
    return f


def build_windows_per_asset(feat, T, H_A, H_B1, H_B5, asset_id):
    X_all   = feat[FEATURE_COLS].values
    har_all = feat[HAR_COLS].values
    log_ret = feat['log_ret'].values
    dates   = feat.index
    n       = len(feat)
    H_MAX   = max(H_A, H_B5)
    out = []
    for i in range(T - 1, n - H_MAX):
        Xw   = X_all[i - T + 1 : i + 1].astype(np.float32)
        harw = har_all[i].astype(np.float32)
        rA   = log_ret[i + 1 : i + 1 + H_A]
        rB1  = log_ret[i + 1 : i + 1 + H_B1]
        rB5  = log_ret[i + 1 : i + 1 + H_B5]
        out.append({
            'X': Xw, 'har': harw,
            'l_A':  float(np.log(np.sqrt(np.mean(rA * rA)) + EPS)),
            'l_B1': float(np.log(np.sqrt(np.mean(rB1 * rB1)) + EPS)),
            'l_B5': float(np.log(np.sqrt(np.mean(rB5 * rB5)) + EPS)),
            'v_A':  float(np.sqrt(np.mean(rA * rA))),
            'date': dates[i], 'asset': asset_id,
        })
    return out


# ==================================================================
# Model
# ==================================================================
class AttentionPool(nn.Module):
    """Per-head attention over the time axis (Def 6.3, 6.4)."""
    def __init__(self, hidden):
        super().__init__()
        self.q = nn.Linear(hidden, 1)

    def forward(self, out):
        w = torch.softmax(self.q(out), dim=1)
        return (w * out).sum(dim=1)


class FiLM(nn.Module):
    """Feature-wise linear modulation. Initialised to identity (γ=1, β=0)."""
    def __init__(self, hidden):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden))
        self.beta  = nn.Parameter(torch.zeros(hidden))

    def forward(self, x):
        return self.gamma * x + self.beta


def softplus_quantiles(z, q_bias=-2.0):
    """
    z: (B, K)
    q[:, 0] = z[:, 0]                        (base, unbounded)
    q[:, k] = q[:, k-1] + softplus(z[:, k] + q_bias)
    Guarantees q[:, 0] < q[:, 1] < ... < q[:, K-1].
    """
    base = z[:, :1]
    inc  = F.softplus(z[:, 1:] + q_bias)        # (B, K-1)
    q    = torch.cat([base, base + torch.cumsum(inc, dim=1)], dim=1)
    return q


class DualHeadTLSTM(nn.Module):
    """
    Head A: K quantiles of ℓ^(H_A).       (Lever A)
    Head B: Gaussian on ℓ^(H_B1), ℓ^(H_B5). (Lever B)
    Per-head attention + FiLM.             (Lever C: 3a, 3c)
    """
    def __init__(self, n_features, hidden=64, num_layers=1,
                 dropout=0.1, k_quantiles=39, q_bias=-2.0):
        super().__init__()
        self.k_quantiles = k_quantiles
        self.q_bias = q_bias
        self.lstm = nn.LSTM(n_features, hidden, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)

        # Lever C.3a: per-head attention pools
        self.pool_a = AttentionPool(hidden)
        self.pool_b = AttentionPool(hidden)

        # Lever C.3c: per-head FiLM
        self.film_a = FiLM(hidden)
        self.film_b = FiLM(hidden)

        # Head A: quantile head (Lever A)
        self.head_a_trunk = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(dropout))
        self.head_a_last  = nn.Linear(64, k_quantiles)

        # Head B: two Gaussian horizons (Lever B)
        self.head_b = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 4))     # (μ1, log σ1, μ5, log σ5)

        self._init_bias()

    def _init_bias(self):
        # Quantile last-layer bias: base at 0, increments at q_bias
        with torch.no_grad():
            self.head_a_last.bias.fill_(self.q_bias)
            self.head_a_last.bias[0] = 0.0

    def forward(self, x, detach_b=False):
        out, _ = self.lstm(x)                     # (B, T, H)
        out = self.drop(out)

        # Head A path
        ha = self.pool_a(out)
        ha = self.film_a(ha)
        za = self.head_a_trunk(ha)
        za = self.head_a_last(za)                 # (B, K)
        q  = softplus_quantiles(za, self.q_bias)  # (B, K) sorted

        # Head B path
        out_b = out.detach() if detach_b else out
        hb = self.pool_b(out_b)
        hb = self.film_b(hb)
        zb = self.head_b(hb)                      # (B, 4)
        mu1, log_s1 = zb[:, 0], zb[:, 1]
        mu5, log_s5 = zb[:, 2], zb[:, 3]
        log_s1 = torch.clamp(log_s1, -6.0, 6.0)
        log_s5 = torch.clamp(log_s5, -6.0, 6.0)
        return q, mu1, log_s1, mu5, log_s5


# ==================================================================
# Losses
# ==================================================================
def pinball_loss(q, y, taus):
    """
    q: (B, K) sorted quantiles
    y: (B,)   target
    taus: (K,) quantile levels
    Returns per-sample pinball loss (B,).
    """
    u = y.unsqueeze(-1) - q                  # (B, K)
    tau = taus.unsqueeze(0)                  # (1, K)
    return torch.max(tau * u, (tau - 1.0) * u)   # (B, K)


def crps_from_quantiles(q, y, taus):
    """Riemann approximation: CRPS ≈ 2 * mean_k ρ_τk(q_k, y)."""
    return 2.0 * pinball_loss(q, y, taus).mean(dim=1)     # (B,)


def crps_gaussian_scalar(y, mu, log_sigma):
    sigma = torch.exp(log_sigma).clamp(min=1e-6)
    z   = (y - mu) / sigma
    phi = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def composite_loss(q, mu1, ls1, mu5, ls5,
                   yA, yB1, yB5, taus):
    """Returns L_A, L_B1, L_B5, L_B."""
    L_A  = crps_from_quantiles(q, yA, taus).mean()
    L_B1 = crps_gaussian_scalar(yB1, mu1, ls1).mean()
    L_B5 = crps_gaussian_scalar(yB5, mu5, ls5).mean()
    L_B  = 0.5 * (L_B1 + L_B5)
    return L_A, L_B1, L_B5, L_B


# ==================================================================
# Running mean (Lever C.3e)
# ==================================================================
class RunningMean:
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.value = None

    def update(self, x):
        x = float(x)
        if self.value is None:
            self.value = x
        else:
            self.value = self.momentum * self.value + (1 - self.momentum) * x
        return self.value


# ==================================================================
# EMA (Lever C in spirit; §13.1)
# ==================================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(),
                                                     alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v.detach())

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


# ==================================================================
# Training
# ==================================================================
def train_model(model, train_loader, val_loader, lam_B,
                epochs, lr, gamma, patience,
                use_ema=True, detach_b=False, warm_t0=40,
                loss_norm=True, verbose=True):
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=gamma)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=warm_t0, T_mult=1, eta_min=lr * 0.05)
    ema = EMA(model, EMA_DECAY) if use_ema else None

    running_A = RunningMean(LOSS_MOMENTUM)
    running_B = RunningMean(LOSS_MOMENTUM)

    taus = torch.tensor(TAU_GRID, device=DEVICE)

    best_val, best_state, wait = float('inf'), None, 0
    history = {'train': [], 'val': [], 'L_A': [], 'L_B': [],
               'rA': [], 'rB': []}

    for ep in range(epochs):
        # --- train ---
        model.train()
        tot = tot_a = tot_b = n = 0.0
        for xb, yA, yB1, yB5 in train_loader:
            xb, yA, yB1, yB5 = (xb.to(DEVICE), yA.to(DEVICE),
                                yB1.to(DEVICE), yB5.to(DEVICE))
            q, mu1, ls1, mu5, ls5 = model(xb, detach_b=detach_b)
            L_A, _, _, L_B = composite_loss(q, mu1, ls1, mu5, ls5,
                                            yA, yB1, yB5, taus)
            # Lever C.3e: normalize by running means
            rA = running_A.update(L_A.item())
            rB = running_B.update(L_B.item())
            if loss_norm and rA > 0 and rB > 0:
                loss = L_A / rA + lam_B * L_B / rB
            else:
                loss = L_A + lam_B * L_B
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema is not None: ema.update(model)
            tot += loss.item() * len(xb)
            tot_a += float(L_A) * len(xb); tot_b += float(L_B) * len(xb)
            n += len(xb)
        train_loss = tot / n
        train_LA = tot_a / n; train_LB = tot_b / n
        history['train'].append(train_loss)
        history['L_A'].append(train_LA)
        history['L_B'].append(train_LB)
        history['rA'].append(rA); history['rB'].append(rB)

        # --- validate on EMA if enabled ---
        backup = None
        if ema is not None:
            backup = {k: v.detach().clone()
                      for k, v in model.state_dict().items()}
            ema.copy_to(model)
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for xb, yA, yB1, yB5 in val_loader:
                xb, yA, yB1, yB5 = (xb.to(DEVICE), yA.to(DEVICE),
                                    yB1.to(DEVICE), yB5.to(DEVICE))
                q, mu1, ls1, mu5, ls5 = model(xb, detach_b=detach_b)
                L_A, _, _, L_B = composite_loss(q, mu1, ls1, mu5, ls5,
                                                yA, yB1, yB5, taus)
                if loss_norm and rA > 0 and rB > 0:
                    loss = L_A / rA + lam_B * L_B / rB
                else:
                    loss = L_A + lam_B * L_B
                tot += loss.item() * len(xb); n += len(xb)
        val_loss = tot / n
        history['val'].append(val_loss)
        if backup is not None:
            model.load_state_dict(backup)
        sched.step()

        improved = val_loss < best_val - 1e-5
        if improved:
            best_val = val_loss
            src = ema.shadow if ema is not None else model.state_dict()
            best_state = {k: v.detach().cpu().clone() for k, v in src.items()}
            wait = 0
        else:
            wait += 1

        if verbose and (ep % 10 == 0 or improved):
            print(f"  ep {ep+1:3d} | "
                  f"train_norm={train_loss:.4f}  "
                  f"(L_A={train_LA:.4f} L_B={train_LB:.4f})  "
                  f"val_norm={val_loss:.4f}{' *' if improved else ''}")
        if wait >= patience:
            if verbose: print(f"  early stop ep {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_val


# ==================================================================
# Inference
# ==================================================================
@torch.no_grad()
def predict(model, X, detach_b=False, bs=2048):
    model.eval()
    Xt = torch.as_tensor(X, dtype=torch.float32)
    qs, m1s, s1s, m5s, s5s = [], [], [], [], []
    for i in range(0, len(Xt), bs):
        xb = Xt[i:i+bs].to(DEVICE)
        q, mu1, ls1, mu5, ls5 = model(xb, detach_b=detach_b)
        qs.append(q.cpu().numpy())
        m1s.append(mu1.cpu().numpy()); s1s.append(torch.exp(ls1).cpu().numpy())
        m5s.append(mu5.cpu().numpy()); s5s.append(torch.exp(ls5).cpu().numpy())
    return (np.concatenate(qs), np.concatenate(m1s), np.concatenate(s1s),
            np.concatenate(m5s), np.concatenate(s5s))


# ==================================================================
# Metrics
# ==================================================================
def qlike(l_true, mu_pred):
    u = l_true - mu_pred
    return float((np.exp(u) - u - 1.0).mean())


def quantile_coverage(y, q, level):
    """Coverage of [q_lo, q_hi] where lo = (1-level)/2, hi = 1 - lo."""
    lo_tau = (1.0 - level) / 2.0
    hi_tau = 1.0 - lo_tau
    idx_lo = int(np.argmin(np.abs(TAU_GRID - lo_tau)))
    idx_hi = int(np.argmin(np.abs(TAU_GRID - hi_tau)))
    lo = q[:, idx_lo]; hi = q[:, idx_hi]
    return float(((y >= lo) & (y <= hi)).mean())


def metrics_from_quantiles(y, q, v_true, label):
    """All metrics derived from the quantile vector q (B, K)."""
    taus = TAU_GRID
    # CRPS via Riemann (using unstandardized y, q)
    q_t = torch.tensor(q, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    tau_t = torch.tensor(taus, dtype=torch.float32)
    crps = float(crps_from_quantiles(q_t, y_t, tau_t).mean())
    # median (τ=0.5) as point forecast
    idx_med = int(np.argmin(np.abs(taus - 0.5)))
    med = q[:, idx_med]
    pearson  = float(np.corrcoef(np.exp(med), v_true)[0, 1])
    spearman = float(pd.Series(med).corr(pd.Series(y), method='spearman'))
    ql       = qlike(y, med)
    mse_log  = float(np.mean((med - y) ** 2))
    cov80    = quantile_coverage(y, q, 0.80)
    cov95    = quantile_coverage(y, q, 0.95)
    print(f"  {label:<30} CRPS={crps:.4f}  ρ={pearson:.4f}  ρ_s={spearman:.4f}  "
          f"QLIKE={ql:.4f}  MSE_ℓ={mse_log:.4f}  cov80={cov80:.3f}  cov95={cov95:.3f}")
    return dict(crps=crps, pearson=pearson, spearman=spearman,
                qlike=ql, mse_log=mse_log, cov80=cov80, cov95=cov95)


def metrics_gaussian(y, mu, sigma, v_true, label):
    """For baselines that are Gaussian (HAR-RV, single-horizon)."""
    crps = float(crps_gaussian_scalar(
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(mu, dtype=torch.float32),
        torch.log(torch.tensor(sigma, dtype=torch.float32).clamp(min=1e-6))
    ).mean())
    pearson  = float(np.corrcoef(np.exp(mu), v_true)[0, 1])
    spearman = float(pd.Series(mu).corr(pd.Series(y), method='spearman'))
    ql       = qlike(y, mu)
    mse_log  = float(np.mean((mu - y) ** 2))
    from scipy.stats import norm
    z80 = norm.ppf(0.90); z95 = norm.ppf(0.975)
    cov80 = float(((y >= mu - z80*sigma) & (y <= mu + z80*sigma)).mean())
    cov95 = float(((y >= mu - z95*sigma) & (y <= mu + z95*sigma)).mean())
    print(f"  {label:<30} CRPS={crps:.4f}  ρ={pearson:.4f}  ρ_s={spearman:.4f}  "
          f"QLIKE={ql:.4f}  MSE_ℓ={mse_log:.4f}  cov80={cov80:.3f}  cov95={cov95:.3f}")
    return dict(crps=crps, pearson=pearson, spearman=spearman,
                qlike=ql, mse_log=mse_log, cov80=cov80, cov95=cov95)


def fit_har_baseline(train_har, train_l, test_har):
    X_tr = np.log(train_har + EPS)
    X_te = np.log(test_har + EPS)
    har = LinearRegression().fit(X_tr, train_l)
    pred_te = har.predict(X_te)
    resid_tr = train_l - har.predict(X_tr)
    sigma = float(resid_tr.std())
    return har, pred_te, sigma


# ==================================================================
# Main
# ==================================================================
def main():
    print("=" * 78)
    print("Dual-Head TLSTM v5.0 — Quantile Head A + Multi-Horizon Head B")
    print("=" * 78)
    print(f"Config:")
    print(f"  T            = {T}")
    print(f"  H_A          = {H_A}  (primary)")
    print(f"  H_B1, H_B5   = {H_B1}, {H_B5}  (auxiliary)")
    print(f"  hidden       = {HIDDEN}")
    print(f"  K quantiles  = {K_QUANTILES}")
    print(f"  λ_B          = {LAMBDA_B}")
    print(f"  loss_norm    = True   (Lever C.3e)")
    print(f"  EMA, warm T0 = {EMA_DECAY}, {WARM_T0}")

    taus_t = torch.tensor(TAU_GRID, device=DEVICE)

    # ---------- [1] Data ----------
    print("\n[1/8] Downloading data...")
    macro = download_macro(START_DATE, END_DATE)
    assets = {}
    for tk in TRAIN_TICKERS:
        raw = download_prices(tk, START_DATE, END_DATE)
        if raw is None: continue
        feat = compute_features(raw, macro)
        if len(feat) < T + H_MAX + 200: continue
        assets[tk] = feat
        print(f"  {tk}: {len(feat)} rows")
    print(f"  Loaded {len(assets)} assets")

    # ---------- [2] Windows ----------
    print("\n[2/8] Building per-asset windows...")
    all_w = []
    for tk, feat in assets.items():
        all_w.extend(build_windows_per_asset(feat, T, H_A, H_B1, H_B5, tk))
    print(f"  Total windows: {len(all_w)}")

    # ---------- [3] Split ----------
    print("\n[3/8] Calendar-date split with embargo T+H_max...")
    master = sorted(assets['SPY'].index)
    m = len(master)
    t1 = int(0.60 * m); t2 = int(0.80 * m)
    emb = T + H_MAX
    train_end_date  = master[t1 - emb - 1]
    val_start_date  = master[t1]
    val_end_date    = master[t2 - emb - 1]
    test_start_date = master[t2]

    train_w = [w for w in all_w if w['date'] <= train_end_date]
    val_w   = [w for w in all_w if val_start_date <= w['date'] <= val_end_date]
    test_w  = [w for w in all_w if w['date'] >= test_start_date]
    print(f"  Train ends {train_end_date.date()}")
    print(f"  Val   in   [{val_start_date.date()}, {val_end_date.date()}]")
    print(f"  Test starts {test_start_date.date()}")
    print(f"  Windows: train={len(train_w)}, val={len(val_w)}, test={len(test_w)}")

    train_X   = np.stack([w['X'] for w in train_w]).astype(np.float32)
    val_X     = np.stack([w['X'] for w in val_w]).astype(np.float32)
    test_X    = np.stack([w['X'] for w in test_w]).astype(np.float32)
    train_har = np.stack([w['har'] for w in train_w]).astype(np.float32)
    test_har  = np.stack([w['har'] for w in test_w]).astype(np.float32)

    def arr(ws, k): return np.array([w[k] for w in ws], dtype=np.float32)
    train_lA, val_lA, test_lA   = arr(train_w,'l_A'),  arr(val_w,'l_A'),  arr(test_w,'l_A')
    train_lB1, val_lB1, test_lB1 = arr(train_w,'l_B1'), arr(val_w,'l_B1'), arr(test_w,'l_B1')
    train_lB5, val_lB5, test_lB5 = arr(train_w,'l_B5'), arr(val_w,'l_B5'), arr(test_w,'l_B5')
    train_vA, test_vA             = arr(train_w,'v_A'), arr(test_w,'v_A')

    # ---------- [4] Target standardization ----------
    print("\n[4/8] Train-fold target standardization...")
    muA = float(train_lA.mean());  sdA = float(train_lA.std()  + 1e-8)
    muB1= float(train_lB1.mean()); sdB1= float(train_lB1.std()+ 1e-8)
    muB5= float(train_lB5.mean()); sdB5= float(train_lB5.std()+ 1e-8)
    print(f"  ℓ^(H_A={H_A}): mean={muA:.4f}  std={train_lA.std():.4f}")
    print(f"  ℓ^(H_B1={H_B1}): mean={muB1:.4f}  std={train_lB1.std():.4f}")
    print(f"  ℓ^(H_B5={H_B5}): mean={muB5:.4f}  std={train_lB5.std():.4f}")

    train_lA_z = (train_lA - muA) / sdA; val_lA_z = (val_lA - muA) / sdA
    test_lA_z  = (test_lA  - muA) / sdA
    train_lB1_z = (train_lB1 - muB1) / sdB1; val_lB1_z = (val_lB1 - muB1) / sdB1
    test_lB1_z  = (test_lB1  - muB1) / sdB1
    train_lB5_z = (train_lB5 - muB5) / sdB5; val_lB5_z = (val_lB5 - muB5) / sdB5
    test_lB5_z  = (test_lB5  - muB5) / sdB5

    mu_f = train_X.mean(axis=(0, 1)).astype(np.float32)
    sd_f = (train_X.std(axis=(0, 1)) + 1e-8).astype(np.float32)
    train_X = (train_X - mu_f) / sd_f
    val_X   = (val_X   - mu_f) / sd_f
    test_X  = (test_X  - mu_f) / sd_f

    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_X),
                      torch.tensor(train_lA_z),
                      torch.tensor(train_lB1_z),
                      torch.tensor(train_lB5_z)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(val_X),
                      torch.tensor(val_lA_z),
                      torch.tensor(val_lB1_z),
                      torch.tensor(val_lB5_z)),
        batch_size=BATCH_SIZE, shuffle=False)

    # ---------- [5] HAR baseline ----------
    print("\n[5/8] Fitting HAR-RV baseline for H_A...")
    har, har_test_lA_pred_z, har_sigma_z = fit_har_baseline(
        train_har, train_lA_z, test_har)
    har_test_lA_pred_o = har_test_lA_pred_z * sdA + muA
    har_sigma_o        = har_sigma_z * sdA
    print(f"  HAR coefs: {har.coef_}, intercept={har.intercept_:.4f}")
    print(f"  HAR train R²: {har.score(np.log(train_har + EPS), train_lA_z):.4f}")

    # ---------- [6] Train main model ----------
    print(f"\n[6/8] Training main Dual-Head TLSTM v5.0...")
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = DualHeadTLSTM(len(FEATURE_COLS), HIDDEN, NUM_LAYERS,
                          DROPOUT, K_QUANTILES, Q_INIT_BIAS).to(DEVICE)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    t0 = time.time()
    hist_main, best_val = train_model(
        model, train_loader, val_loader, LAMBDA_B,
        EPOCHS, LR, GAMMA, PATIENCE,
        use_ema=True, detach_b=False, warm_t0=WARM_T0, loss_norm=True)
    print(f"  Time: {(time.time()-t0)/60:.1f} min | best val_norm = {best_val:.4f}")

    # ---------- Test predictions ----------
    q_test_z, tm1_z, ts1, tm5_z, ts5 = predict(model, test_X)
    q_val_z,  vm1_z, vs1, vm5_z, vs5 = predict(model, val_X)

    # Unstandardize
    q_test_o = q_test_z * sdA + muA
    q_val_o  = q_val_z  * sdA + muA
    # Head B unscale
    ts1_o = ts1 * sdB1; ts5_o = ts5 * sdB5
    tm1_o = tm1_z * sdB1 + muB1; tm5_o = tm5_z * sdB5 + muB5

    # ---------- [7] Baselines ----------
    print("\n  Training single-head baseline (quantile, λ_B=0)...")
    torch.manual_seed(SEED); np.random.seed(SEED)
    m_sh = DualHeadTLSTM(len(FEATURE_COLS), HIDDEN, NUM_LAYERS,
                         DROPOUT, K_QUANTILES, Q_INIT_BIAS).to(DEVICE)
    train_model(m_sh, train_loader, val_loader, 0.0,
                EPOCHS, LR, GAMMA, PATIENCE,
                use_ema=True, detach_b=False, warm_t0=WARM_T0,
                loss_norm=False, verbose=True)
    q_sh_test_z, _, _, _, _ = predict(m_sh, test_X)
    q_sh_test_o = q_sh_test_z * sdA + muA

    print("  Training detached Head B baseline...")
    torch.manual_seed(SEED); np.random.seed(SEED)
    m_dt = DualHeadTLSTM(len(FEATURE_COLS), HIDDEN, NUM_LAYERS,
                         DROPOUT, K_QUANTILES, Q_INIT_BIAS).to(DEVICE)
    train_model(m_dt, train_loader, val_loader, LAMBDA_B,
                EPOCHS, LR, GAMMA, PATIENCE,
                use_ema=True, detach_b=True, warm_t0=WARM_T0,
                loss_norm=True, verbose=True)
    q_dt_test_z, _, _, _, _ = predict(m_dt, test_X, detach_b=True)
    q_dt_test_o = q_dt_test_z * sdA + muA

    # Constant baseline: median = train mean, ±2σ for intervals
    const_med = np.full(len(test_lA), muA)
    # Build a Gaussian quantile vector for the constant predictor
    from scipy.stats import norm
    const_q = np.zeros((len(test_lA), K_QUANTILES), dtype=np.float32)
    for k, tau in enumerate(TAU_GRID):
        const_q[:, k] = muA + norm.ppf(tau) * float(train_lA.std())

    # ---------- [8] Evaluate ----------
    print("\n[8/8] Test metrics (Head A primary H={})".format(H_A))
    print("=" * 118)
    print(f"{'Model':<30} {'CRPS':>8} {'Pearson':>9} {'Spear':>8} "
          f"{'QLIKE':>8} {'MSE_ℓ':>8} {'cov80':>7} {'cov95':>7}")
    print("-" * 118)
    res = {}
    res['Main dual-head']    = metrics_from_quantiles(test_lA, q_test_o, test_vA, 'Main dual-head')
    res['Single-head']       = metrics_from_quantiles(test_lA, q_sh_test_o, test_vA, 'Single-head')
    res['Detached Head B']   = metrics_from_quantiles(test_lA, q_dt_test_o, test_vA, 'Detached Head B')
    res['HAR-RV']            = metrics_gaussian(test_lA, har_test_lA_pred_o,
                                                np.full(len(test_lA), har_sigma_o),
                                                test_vA, 'HAR-RV')
    res['Constant']          = metrics_from_quantiles(test_lA, const_q, test_vA, 'Constant')
    print("=" * 118)

    # ---------- Auxiliary horizons (Head B) ----------
    print(f"\nAuxiliary head diagnostics:")
    for h, m, s, ly in [(H_B1, tm1_o, ts1_o, test_lB1),
                        (H_B5, tm5_o, ts5_o, test_lB5)]:
        crps_h = float(crps_gaussian_scalar(
            torch.tensor(ly, dtype=torch.float32),
            torch.tensor(m, dtype=torch.float32),
            torch.log(torch.tensor(s, dtype=torch.float32).clamp(min=1e-6))
        ).mean())
        rho_h = float(np.corrcoef(np.exp(m), np.exp(ly))[0, 1])
        print(f"  H_B={h:>2}  CRPS={crps_h:.4f}  ρ={rho_h:.4f}")

    # ---------- Encoder-separation verdict ----------
    print("\n" + "-" * 70)
    print("ENCODER-SEPARATION TEST — Main vs Single-head")
    print("-" * 70)
    base_crps = res['Single-head']['crps']
    for name in ['Main dual-head', 'Detached Head B', 'HAR-RV', 'Constant']:
        dc = res[name]['crps'] - base_crps
        rel = 100.0 * dc / base_crps if base_crps > 0 else 0
        verdict = ''
        if name == 'Main dual-head':
            if rel < -2.0:   verdict = '  PASS — dual-head ≥2% better'
            elif rel < -0.3: verdict = '  ~ marginal (<2%)'
            else:            verdict = '  FAIL — no separation'
        print(f"  {name:<20} ΔCRPS={dc:+.4f} ({rel:+.2f}%){verdict}")

    # ---------- σ̂ separation diagnostic (quantile-based) ----------
    print("\nσ̂-width decile diagnostic (Head A interval width at 80%):")
    widths = q_test_o[:, int(np.argmin(np.abs(TAU_GRID - 0.90)))] \
           - q_test_o[:, int(np.argmin(np.abs(TAU_GRID - 0.10)))]
    order  = np.argsort(widths)
    deciles = np.array_split(order, 10)
    dec_crps = []
    for d in deciles:
        c = float(crps_from_quantiles(
            torch.tensor(q_test_o[d], dtype=torch.float32),
            torch.tensor(test_lA[d], dtype=torch.float32),
            torch.tensor(TAU_GRID, dtype=torch.float32)).mean())
        dec_crps.append(c)
    print(f"  Narrowest decile CRPS: {dec_crps[0]:.4f}")
    print(f"  Widest   decile CRPS: {dec_crps[-1]:.4f}")
    if dec_crps[0] < dec_crps[-1]:
        print(f"  ✓ Interval width separates easy from hard windows")

    # ---------- Save ----------
    torch.save({
        'state_dict': model.state_dict(),
        'config': {
            'n_features': len(FEATURE_COLS), 'hidden': HIDDEN,
            'num_layers': NUM_LAYERS, 'dropout': DROPOUT,
            'T': T, 'H_A': H_A, 'H_B1': H_B1, 'H_B5': H_B5,
            'k_quantiles': K_QUANTILES, 'tau_grid': TAU_GRID.tolist(),
            'q_bias': Q_INIT_BIAS, 'lambda_B': LAMBDA_B,
            'ema_decay': EMA_DECAY,
        },
        'feature_cols': FEATURE_COLS,
        'mu_f': mu_f, 'sd_f': sd_f,
        'muA': muA, 'sdA': sdA,
        'muB1': muB1, 'sdB1': sdB1,
        'muB5': muB5, 'sdB5': sdB5,
        'har_coef': har.coef_, 'har_intercept': har.intercept_,
        'har_sigma_o': har_sigma_o,
        'history': hist_main,
        'test_metrics': {k: {kk: float(vv) for kk, vv in v.items()}
                         for k, v in res.items()},
    }, CKPT_PATH)
    print(f"\nSaved: {CKPT_PATH}")

    # ---------- Plots ----------
    print("\nPlotting...")
    fig = plt.figure(figsize=(20, 12))
    gs  = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)

    # 1. Loss curves
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(hist_main['train'], label='total train (normalized)')
    ax.plot(hist_main['val'],   label='total val (normalized)')
    ax.plot(hist_main['L_A'],   label='L_A raw', alpha=0.6)
    ax.plot(hist_main['L_B'],   label='L_B raw', alpha=0.6)
    ax.set(xlabel='epoch', ylabel='loss', title='Loss curves (v5.0)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 2. Quantile fan at a sample point
    ax = fig.add_subplot(gs[0, 1])
    sample_idx = np.random.RandomState(0).choice(len(test_lA), 200, replace=False)
    for i in sample_idx[:30]:
        ax.plot(TAU_GRID, q_test_o[i], color='steelblue', alpha=0.3, lw=0.5)
    ax.plot(TAU_GRID, const_q[0], 'k--', lw=1.5, label='Constant baseline')
    ax.set(xlabel='τ quantile level', ylabel='quantile value (log-vol)',
           title='Quantile fan (30 test samples)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 3. Coverage from quantiles
    ax = fig.add_subplot(gs[0, 2])
    levels = np.linspace(0.05, 0.95, 19)
    covs_main = [quantile_coverage(test_lA, q_test_o, lv) for lv in levels]
    covs_sh   = [quantile_coverage(test_lA, q_sh_test_o, lv) for lv in levels]
    covs_const= [quantile_coverage(test_lA, const_q, lv) for lv in levels]
    ax.plot(levels, covs_main, 's-',  label='Main dual-head')
    ax.plot(levels, covs_sh,   'o--', label='Single-head')
    ax.plot(levels, covs_const,'x:',  label='Constant')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Ideal')
    ax.set(xlabel='Nominal coverage', ylabel='Empirical coverage',
           title='Reliability curve (Head A, from quantiles)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 4. CRPS by interval-width decile
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(range(1, 11), dec_crps, color='tab:purple')
    ax.set(xlabel='Interval-width decile (1=narrow)',
           ylabel='CRPS', title='CRPS vs σ̂-width decile')
    ax.grid(alpha=0.3)

    # 5. Model comparison table
    ax = fig.add_subplot(gs[1, 1]); ax.axis('off')
    rows = []
    for name in ['Main dual-head', 'Single-head', 'Detached Head B',
                 'HAR-RV', 'Constant']:
        r = res[name]
        rows.append([name, f"{r['crps']:.4f}", f"{r['pearson']:.4f}",
                     f"{r['qlike']:.4f}", f"{r['cov80']:.3f}"])
    cols = ['Model', 'CRPS', 'ρ', 'QLIKE', 'cov80']
    tbl = ax.table(cellText=rows, colLabels=cols,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.1, 1.5)
    ax.set_title(f'Test metrics (H_A={H_A})', pad=10)

    # 6. Per-asset Pearson
    ax = fig.add_subplot(gs[1, 2])
    asset_ids = [w['asset'] for w in test_w]
    pers = []
    for a in sorted(set(asset_ids)):
        m = np.array(asset_ids) == a
        med = q_test_o[m, int(np.argmin(np.abs(TAU_GRID - 0.5)))]
        p = float(np.corrcoef(np.exp(med), test_vA[m])[0, 1])
        pers.append((a, p))
    pers.sort(key=lambda x: x[1])
    names = [p[0] for p in pers]; vals = [p[1] for p in pers]
    colors = ['tab:red' if v < 0.5 else 'tab:green' for v in vals]
    ax.barh(names, vals, color=colors)
    ax.axvline(0.5, ls='--', color='gray')
    ax.set(xlabel=f'Pearson ρ(exp(median), v^(H={H_A}))',
           title='Per-asset Pearson correlation')
    ax.grid(alpha=0.3)

    plt.suptitle(
        f'Dual-Head TLSTM v5.0 — quantile Head A (K={K_QUANTILES}) + '
        f'multi-horizon Head B ({H_B1},{H_B5}), primary H_A={H_A}',
        fontsize=13, y=0.995)
    plt.savefig('dual_head_tlstm_v5.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("Saved: dual_head_tlstm_v5.png")


if __name__ == "__main__":
    main()