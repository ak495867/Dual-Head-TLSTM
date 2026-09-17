"""
Dual-Head TLSTM v4.0 — spec-compliant implementation.
======================================================
Primary head A: log realized volatility at H_A = 20
Auxiliary head B: log realized volatility at H_B = 1
Both distributional, both CRPS-trained.

Theories implemented:
  §10  Encoder-separation criterion (Def 10.7, Thm 10.8, Thm 10.12)
       Multi-horizon is a case (H) separation.
  §12  Interval construction + temperature scaling (Def 12.5, Prop 12.6)
  §13  EMA, warm restarts, per-head attention pooling
  §15  Full baseline suite and reportable-claims protocol

Causality (unchanged):
  §2.4  Calendar-date split with embargo T + H_max
  §2.5  Per-asset windows
  §4.1  Train-fold-only feature standardization
  §4.2  Train-fold-only target standardization

Saves: dual_head_tlstm_v4.pth
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
from scipy.stats import norm
from scipy.optimize import minimize_scalar
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
# §0  Config
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

# --- Model ---
T          = 30               # §5 window length
H_A        = 20               # §1.3 primary horizon (regime-driven)
H_B        = 1                # §1.3 auxiliary horizon (activity-driven)
H_MAX      = max(H_A, H_B)

HIDDEN     = 64               # §6.1 hidden width
NUM_LAYERS = 1
DROPOUT    = 0.1
C_SIGMA    = 6.0              # §7.2 log-σ clamp

# --- Training ---
BATCH_SIZE = 512
EPOCHS     = 150              # §14.5 early-stop ceiling
LR         = 5e-4             # §14.1 η_0
GAMMA      = 1e-5             # §14.1 AdamW decoupled decay
LAMBDA_B   = 0.5              # §9.5 auxiliary loss weight (scale-matched)
PATIENCE   = 20               # §14.5

EMA_DECAY  = 0.999            # §13.1
WARM_RESTART_T0 = 40          # §13.2 period

# --- Targets ---
EPS        = 1e-6

# --- Paths ---
CKPT_PATH  = 'dual_head_tlstm_v4.pth'
CACHE_DIR  = './cache_v4'
os.makedirs(CACHE_DIR, exist_ok=True)

# ==================================================================
# §3  Feature set (16 causal features, Def 3.8 extended)
# ==================================================================
FEATURE_COLS = [
    'r1','r5','r20',
    'vol20','vol60',
    'ma5_ratio','ma20_ratio','rsi',
    'vol_pct','vol_imbalance',
    'gap','intraday_range',
    'vix','vix_pct','tnx_pct',
    'vix_term',
]
HAR_COLS = ['rv_d','rv_w','rv_m']    # for §15.3 baseline only


# ==================================================================
# §3  Data download
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
    # ^VIX3M for term structure
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


# ==================================================================
# §3  Feature construction
# ==================================================================
def compute_features(df, macro):
    """
    §3.1–3.7. Every coordinate is F_t-measurable (Prop 3.9).
    Macro features use forward-fill only (Prop 2.6).
    """
    close  = df['Close'].astype(float)
    open_  = df['Open'].astype(float)
    high   = df['High'].astype(float)
    low    = df['Low'].astype(float)
    volume = df['Volume'].astype(float)
    logc   = np.log(close)
    f = pd.DataFrame(index=df.index)

    # §3.1 returns
    f['r1']  = logc.diff(1)
    f['r5']  = logc.diff(5)
    f['r20'] = logc.diff(20)

    # §3.2 realized vol
    f['vol20'] = f['r1'].rolling(20).std()
    f['vol60'] = f['r1'].rolling(60).std()

    # §3.3 MA ratios
    f['ma5_ratio']  = close / close.rolling(5).mean()  - 1.0
    f['ma20_ratio'] = close / close.rolling(20).mean() - 1.0

    # §3.4 Wilder RSI (n=14)
    delta = close.diff()
    U = delta.clip(lower=0); D = (-delta).clip(lower=0)
    n = 14
    Ub = U.ewm(alpha=1.0/n, adjust=False).mean()
    Db = D.ewm(alpha=1.0/n, adjust=False).mean()
    f['rsi'] = 100.0 * (1.0 - 1.0 / (1.0 + Ub / (Db + 1e-9)))

    # §3.5 relative volume
    f['vol_pct'] = volume / (volume.rolling(20).mean() + 1e-9) - 1.0

    # §3.7 volume imbalance (MAD-normalized)
    vmed = volume.rolling(20).median()
    vmad = (volume - vmed).abs().rolling(20).median() + 1e-6
    f['vol_imbalance'] = (volume - vmed) / vmad

    # §3.7 overnight gap and intraday range
    f['gap']            = np.log(open_ / close.shift(1) + 1e-9)
    f['intraday_range'] = np.log(high / (low + 1e-9))

    # §3.6 macro (forward-fill only)
    macro_ff = macro.reindex(f.index, method='ffill')
    f['vix']     = macro_ff['vix']
    f['vix_pct'] = macro_ff['vix'].pct_change(1)
    f['tnx_pct'] = macro_ff['tnx'].pct_change(1)

    # §3.7 term spread
    if 'vix3m' in macro_ff.columns and macro_ff['vix3m'].notna().sum() > 100:
        f['vix_term'] = macro_ff['vix'] - macro_ff['vix3m']
    else:
        f['vix_term'] = 0.0

    # HAR features (used only by §15.3 baseline, not fed to LSTM)
    r2 = f['r1'] ** 2
    f['rv_d'] = r2
    f['rv_w'] = r2.rolling(5).mean()
    f['rv_m'] = r2.rolling(22).mean()

    # for target construction
    f['log_ret'] = f['r1']

    f = f.replace([np.inf, -np.inf], np.nan).dropna()
    return f


# ==================================================================
# §5  Per-asset windowing (Spec 2.13, Prop 2.12)
# ==================================================================
def build_windows_per_asset(feat, T, H_A, H_B, asset_id):
    """
    X_{a,t} = feat[i-T+1 : i+1]          (T, d)
    ℓ^{(H_A)} = log sqrt( mean_{1..H_A} r^2 )
    ℓ^{(H_B)} = log sqrt( mean_{1..H_B} r^2 )
    """
    X_all   = feat[FEATURE_COLS].values
    har_all = feat[HAR_COLS].values
    log_ret = feat['log_ret'].values
    dates   = feat.index
    n       = len(feat)
    H_MAX   = max(H_A, H_B)
    out = []
    for i in range(T - 1, n - H_MAX):
        Xw    = X_all[i - T + 1 : i + 1].astype(np.float32)
        harw  = har_all[i].astype(np.float32)
        r_a   = log_ret[i + 1 : i + 1 + H_A]
        r_b   = log_ret[i + 1 : i + 1 + H_B]
        v_a   = float(np.sqrt(np.mean(r_a * r_a)))
        v_b   = float(np.sqrt(np.mean(r_b * r_b)))
        out.append({
            'X': Xw, 'har': harw,
            'v_A': v_a, 'v_B': v_b,
            'l_A': float(np.log(v_a + EPS)),
            'l_B': float(np.log(v_b + EPS)),
            'date': dates[i], 'asset': asset_id,
        })
    return out


# ==================================================================
# §6  Encoder with per-head attention pooling (Def 6.3, 6.4)
# ==================================================================
class AttentionPool(nn.Module):
    """α_i = softmax(q^T h_i); ψ = Σ α_i h_i. Def 6.3."""
    def __init__(self, hidden):
        super().__init__()
        self.q = nn.Linear(hidden, 1)

    def forward(self, out):                       # (B, T, H)
        w = torch.softmax(self.q(out), dim=1)     # (B, T, 1)
        return (w * out).sum(dim=1)               # (B, H)


class DualHeadTLSTM(nn.Module):
    """
    §6.1 shared LSTM. §6.4 per-head attention pooling.
    §7.2 Head A: Gaussian on ℓ^{(H_A)}.
    §8.1 Head B: Gaussian on ℓ^{(H_B)}.
    """
    def __init__(self, n_features, hidden=64, num_layers=1,
                 dropout=0.1, c_sigma=6.0):
        super().__init__()
        self.c_sigma = c_sigma
        self.lstm = nn.LSTM(n_features, hidden, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)

        # Per-head readouts (Def 6.4)
        self.pool_a = AttentionPool(hidden)
        self.pool_b = AttentionPool(hidden)

        # §7.2 Head A: (μ_A, log σ_A)
        self.head_a = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, 2))
        # §8.1 Head B: (μ_B, log σ_B)
        self.head_b = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, 2))

    def forward(self, x, detach_b=False):
        out, _ = self.lstm(x)                     # (B, T, H)
        out = self.drop(out)

        ha = self.pool_a(out)
        a  = self.head_a(ha)
        mu_a    = a[:, 0]
        log_s_a = torch.clamp(a[:, 1], -self.c_sigma, self.c_sigma)

        hb_in = out.detach() if detach_b else out
        hb = self.pool_b(hb_in)
        b  = self.head_b(hb)
        mu_b    = b[:, 0]
        log_s_b = torch.clamp(b[:, 1], -self.c_sigma, self.c_sigma)

        return mu_a, log_s_a, mu_b, log_s_b


# ==================================================================
# §9  Losses
# ==================================================================
def crps_gaussian_scalar(y, mu, log_sigma):
    """§9.2 Gaussian CRPS closed form."""
    sigma = torch.exp(log_sigma).clamp(min=1e-6)
    z   = (y - mu) / sigma
    phi = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def composite_loss(mu_a, ls_a, mu_b, ls_b, y_a, y_b, lam_B):
    """§9.5  L_total = L_A + λ_B · L_B.  λ_C = 0 (multi-horizon is not functional)."""
    L_A = crps_gaussian_scalar(y_a, mu_a, ls_a).mean()
    L_B = crps_gaussian_scalar(y_b, mu_b, ls_b).mean()
    total = L_A + lam_B * L_B
    return total, float(L_A), float(L_B)


# ==================================================================
# §13.1  EMA
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
# §14  Training
# ==================================================================
def train_model(model, train_loader, val_loader, lam_B,
                epochs, lr, gamma, patience,
                use_ema=True, detach_b=False,
                warm_restart_t0=40):
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=gamma)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=warm_restart_t0, T_mult=1, eta_min=lr * 0.05)

    ema = EMA(model, EMA_DECAY) if use_ema else None

    best_val, best_state, wait = float('inf'), None, 0
    history = {'train': [], 'val': [], 'L_A': [], 'L_B': []}

    for ep in range(epochs):
        # --- train ---
        model.train()
        tot = tot_a = tot_b = n = 0.0
        for xb, ya, yb in train_loader:
            xb, ya, yb = xb.to(DEVICE), ya.to(DEVICE), yb.to(DEVICE)
            mu_a, ls_a, mu_b, ls_b = model(xb, detach_b=detach_b)
            loss, la, lb = composite_loss(mu_a, ls_a, mu_b, ls_b,
                                          ya, yb, lam_B)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema is not None:
                ema.update(model)
            tot += loss.item() * len(xb)
            tot_a += la * len(xb); tot_b += lb * len(xb); n += len(xb)
        train_loss = tot / n
        train_LA = tot_a / n; train_LB = tot_b / n
        history['train'].append(train_loss)
        history['L_A'].append(train_LA)
        history['L_B'].append(train_LB)

        # --- validate on EMA if enabled ---
        backup = None
        if ema is not None:
            backup = {k: v.detach().clone()
                      for k, v in model.state_dict().items()}
            ema.copy_to(model)
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for xb, ya, yb in val_loader:
                xb, ya, yb = xb.to(DEVICE), ya.to(DEVICE), yb.to(DEVICE)
                mu_a, ls_a, mu_b, ls_b = model(xb, detach_b=detach_b)
                loss, _, _ = composite_loss(mu_a, ls_a, mu_b, ls_b,
                                            ya, yb, lam_B)
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

        if ep % 10 == 0 or improved:
            print(f"  ep {ep+1:3d} | train={train_loss:.4f} "
                  f"(LA={train_LA:.4f} LB={train_LB:.4f}) "
                  f"val={val_loss:.4f}{' *' if improved else ''}")
        if wait >= patience:
            print(f"  early stop ep {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_val


# ==================================================================
# Inference
# ==================================================================
@torch.no_grad()
def predict(model, X, detach_b=False, bs=2048):
    """Returns (μ_A, σ_A, μ_B, σ_B) all in standardized target units."""
    model.eval()
    Xt = torch.as_tensor(X, dtype=torch.float32)
    mu_as, sig_as, mu_bs, sig_bs = [], [], [], []
    for i in range(0, len(Xt), bs):
        xb = Xt[i:i+bs].to(DEVICE)
        mu_a, ls_a, mu_b, ls_b = model(xb, detach_b=detach_b)
        mu_as.append(mu_a.cpu().numpy())
        sig_as.append(torch.exp(ls_a).cpu().numpy())
        mu_bs.append(mu_b.cpu().numpy())
        sig_bs.append(torch.exp(ls_b).cpu().numpy())
    return (np.concatenate(mu_as), np.concatenate(sig_as),
            np.concatenate(mu_bs), np.concatenate(sig_bs))


# ==================================================================
# §15  Metrics
# ==================================================================
def qlike(l_true, mu_pred):
    u = l_true - mu_pred
    return float((np.exp(u) - u - 1.0).mean())


def gauss_coverage(l_true, mu, sigma, level):
    z = norm.ppf(0.5 + level / 2.0)
    lo, hi = mu - z * sigma, mu + z * sigma
    return float(((l_true >= lo) & (l_true <= hi)).mean())


def full_metrics(l_true, mu_pred, sigma_pred, v_true, label):
    """§15.1 metrics suite."""
    crps = float(crps_gaussian_scalar(
        torch.tensor(l_true, dtype=torch.float32),
        torch.tensor(mu_pred, dtype=torch.float32),
        torch.log(torch.tensor(sigma_pred, dtype=torch.float32).clamp(min=1e-6))
    ).mean())
    pearson  = float(np.corrcoef(np.exp(mu_pred), v_true)[0, 1])
    spearman = float(pd.Series(mu_pred).corr(pd.Series(l_true),
                                             method='spearman'))
    ql       = qlike(l_true, mu_pred)
    mse_log  = float(np.mean((mu_pred - l_true) ** 2))
    cov80    = gauss_coverage(l_true, mu_pred, sigma_pred, 0.80)
    cov95    = gauss_coverage(l_true, mu_pred, sigma_pred, 0.95)
    print(f"  {label:<28} CRPS={crps:.4f}  ρ={pearson:.4f}  ρ_s={spearman:.4f}  "
          f"QLIKE={ql:.4f}  MSE_ℓ={mse_log:.4f}  cov80={cov80:.3f}  cov95={cov95:.3f}")
    return dict(crps=crps, pearson=pearson, spearman=spearman,
                qlike=ql, mse_log=mse_log, cov80=cov80, cov95=cov95)


# ==================================================================
# §12.2  Temperature scaling
# ==================================================================
def fit_sigma_temperature(mu_val, sig_val, l_val):
    """
    §12.2. Minimize squared coverage error over {80%, 95%} on validation.
    """
    def loss(t):
        s = sig_val * t
        return (gauss_coverage(l_val, mu_val, s, 0.80) - 0.80) ** 2 \
             + (gauss_coverage(l_val, mu_val, s, 0.95) - 0.95) ** 2
    res = minimize_scalar(loss, bounds=(0.5, 2.5), method='bounded')
    return float(res.x)


# ==================================================================
# §15.3  HAR-RV baseline
# ==================================================================
def fit_har_baseline(train_har, train_l, test_har):
    """§15.3 Corsi 2009. Log-RV regression on daily/weekly/monthly RV."""
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
    print("Dual-Head TLSTM v4.0 — encoder-separation test")
    print("=" * 78)
    print(f"Config:")
    print(f"  T          = {T}")
    print(f"  H_A        = {H_A}  (primary: regime-driven)")
    print(f"  H_B        = {H_B}  (auxiliary: activity-driven)")
    print(f"  hidden     = {HIDDEN}, layers = {NUM_LAYERS}")
    print(f"  λ_B        = {LAMBDA_B}")
    print(f"  EMA decay  = {EMA_DECAY}")
    print(f"  Warm T0    = {WARM_RESTART_T0}")

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
        all_w.extend(build_windows_per_asset(feat, T, H_A, H_B, tk))
    print(f"  Total windows: {len(all_w)}")

    # ---------- [3] Calendar-date split (§2.4) ----------
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

    train_X    = np.stack([w['X']    for w in train_w]).astype(np.float32)
    val_X      = np.stack([w['X']    for w in val_w]).astype(np.float32)
    test_X     = np.stack([w['X']    for w in test_w]).astype(np.float32)
    train_har  = np.stack([w['har']  for w in train_w]).astype(np.float32)
    test_har   = np.stack([w['har']  for w in test_w]).astype(np.float32)

    train_lA   = np.array([w['l_A'] for w in train_w], dtype=np.float32)
    val_lA     = np.array([w['l_A'] for w in val_w],   dtype=np.float32)
    test_lA    = np.array([w['l_A'] for w in test_w],  dtype=np.float32)

    train_lB   = np.array([w['l_B'] for w in train_w], dtype=np.float32)
    val_lB     = np.array([w['l_B'] for w in val_w],   dtype=np.float32)
    test_lB    = np.array([w['l_B'] for w in test_w],  dtype=np.float32)

    train_vA   = np.array([w['v_A'] for w in train_w], dtype=np.float32)
    test_vA    = np.array([w['v_A'] for w in test_w],  dtype=np.float32)
    train_vB   = np.array([w['v_B'] for w in train_w], dtype=np.float32)
    test_vB    = np.array([w['v_B'] for w in test_w],  dtype=np.float32)

    # ---------- [4] Target standardization (§4.2) ----------
    print("\n[4/8] Train-fold target standardization...")
    mu_lA = float(train_lA.mean()); sd_lA = float(train_lA.std() + 1e-8)
    mu_lB = float(train_lB.mean()); sd_lB = float(train_lB.std() + 1e-8)
    print(f"  ℓ^(H_A={H_A}): mean={mu_lA:.4f} std={train_lA.std():.4f}")
    print(f"  ℓ^(H_B={H_B}):  mean={mu_lB:.4f} std={train_lB.std():.4f}")

    train_lA_z = (train_lA - mu_lA) / sd_lA
    val_lA_z   = (val_lA   - mu_lA) / sd_lA
    test_lA_z  = (test_lA  - mu_lA) / sd_lA
    train_lB_z = (train_lB - mu_lB) / sd_lB
    val_lB_z   = (val_lB   - mu_lB) / sd_lB
    test_lB_z  = (test_lB  - mu_lB) / sd_lB

    # Feature scaler (train-fold-only, Lemma 4.2)
    mu_f = train_X.mean(axis=(0, 1)).astype(np.float32)
    sd_f = (train_X.std(axis=(0, 1)) + 1e-8).astype(np.float32)
    train_X = (train_X - mu_f) / sd_f
    val_X   = (val_X   - mu_f) / sd_f
    test_X  = (test_X  - mu_f) / sd_f

    # ---------- Loaders ----------
    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_X),
                      torch.tensor(train_lA_z),
                      torch.tensor(train_lB_z)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(val_X),
                      torch.tensor(val_lA_z),
                      torch.tensor(val_lB_z)),
        batch_size=BATCH_SIZE, shuffle=False)

    # ---------- [5] HAR-RV baseline (§15.3) ----------
    print("\n[5/8] Fitting HAR-RV baseline (§15.3)...")
    har, har_test_lA_pred, har_sigma_z = fit_har_baseline(
        train_har, train_lA_z, test_har)
    # In original units, for the metrics table
    har_test_lA_pred_o = har_test_lA_pred * sd_lA + mu_lA
    har_sigma_o        = har_sigma_z * sd_lA
    print(f"  HAR coefs: {har.coef_}, intercept={har.intercept_:.4f}")
    print(f"  HAR train R²: {har.score(np.log(train_har + EPS), train_lA_z):.4f}")

    # ---------- [6] Train main model ----------
    print(f"\n[6/8] Training main Dual-Head TLSTM v4.0...")
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = DualHeadTLSTM(len(FEATURE_COLS), HIDDEN, NUM_LAYERS,
                          DROPOUT, C_SIGMA).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")
    t0 = time.time()
    hist_main, best_val = train_model(
        model, train_loader, val_loader, LAMBDA_B,
        EPOCHS, LR, GAMMA, PATIENCE,
        use_ema=True, detach_b=False,
        warm_restart_t0=WARM_RESTART_T0)
    print(f"  Time: {(time.time()-t0)/60:.1f} min | best val = {best_val:.4f}")

    # ---------- [7] σ-temperature scaling (§12.2) ----------
    print("\n[7/8] Fitting σ-calibration temperatures on validation...")
    val_muA, val_sigA, val_muB, val_sigB = predict(model, val_X)
    # validation metrics in standardized units for the calibration objective
    t_A = fit_sigma_temperature(val_muA, val_sigA, val_lA_z)
    t_B = fit_sigma_temperature(val_muB, val_sigB, val_lB_z)
    print(f"  t_A (H={H_A}) = {t_A:.4f}")
    print(f"  t_B (H={H_B})  = {t_B:.4f}")

    # ---------- Test predictions ----------
    te_muA_z, te_sigA_z, te_muB_z, te_sigB_z = predict(model, test_X)
    # unstandardize
    te_muA_o  = te_muA_z * sd_lA + mu_lA
    te_sigA_o = te_sigA_z * sd_lA
    te_muB_o  = te_muB_z * sd_lB + mu_lB
    te_sigB_o = te_sigB_z * sd_lB
    te_sigA_cal = te_sigA_o * t_A
    te_sigB_cal = te_sigB_o * t_B

    # ---------- Baselines (§15.2) ----------
    print("\n  Training single-head baseline (λ_B=0)...")
    torch.manual_seed(SEED); np.random.seed(SEED)
    m_sh = DualHeadTLSTM(len(FEATURE_COLS), HIDDEN, NUM_LAYERS,
                         DROPOUT, C_SIGMA).to(DEVICE)
    train_model(m_sh, train_loader, val_loader, 0.0,
                EPOCHS, LR, GAMMA, PATIENCE,
                use_ema=True, detach_b=False,
                warm_restart_t0=WARM_RESTART_T0)
    sh_muA_z, sh_sigA_z, _, _ = predict(m_sh, test_X)
    sh_muA_o  = sh_muA_z * sd_lA + mu_lA
    sh_sigA_o = sh_sigA_z * sd_lA
    _, sh_valSig_z, _, _ = predict(m_sh, val_X)
    t_sh = fit_sigma_temperature(sh_muA_z if False else
                                 (sh_muA_z * 1.0), sh_valSig_z, val_lA_z)
    sh_sigA_cal = sh_sigA_o * t_sh

    print("  Training detached Head B baseline...")
    torch.manual_seed(SEED); np.random.seed(SEED)
    m_dt = DualHeadTLSTM(len(FEATURE_COLS), HIDDEN, NUM_LAYERS,
                         DROPOUT, C_SIGMA).to(DEVICE)
    train_model(m_dt, train_loader, val_loader, LAMBDA_B,
                EPOCHS, LR, GAMMA, PATIENCE,
                use_ema=True, detach_b=True,
                warm_restart_t0=WARM_RESTART_T0)
    dt_muA_z, dt_sigA_z, _, _ = predict(m_dt, test_X, detach_b=True)
    dt_muA_o  = dt_muA_z * sd_lA + mu_lA
    dt_sigA_o = dt_sigA_z * sd_lA
    _, dt_valSig_z, _, _ = predict(m_dt, val_X, detach_b=True)
    t_dt = fit_sigma_temperature(dt_muA_z, dt_valSig_z, val_lA_z)
    dt_sigA_cal = dt_sigA_o * t_dt

    # Constant baseline (§15.2 Def 15.2)
    const_mu  = np.full(len(test_lA), mu_lA)
    const_sig = np.full(len(test_lA), float(train_lA.std()))

    # ---------- [8] Evaluate ----------
    print("\n[8/8] Test metrics (§15.1):")
    print("=" * 112)
    print(f"{'Model (primary H=' + str(H_A) + ')':<34} {'CRPS':>8} "
          f"{'Pearson':>9} {'Spear':>8} {'QLIKE':>8} {'MSE_ℓ':>8} "
          f"{'cov80':>7} {'cov95':>7}")
    print("-" * 112)

    res = {}
    res['Main dual-head (cal)']   = full_metrics(
        test_lA, te_muA_o, te_sigA_cal, test_vA, 'Main dual-head (cal)')
    res['Main dual-head (raw)']   = full_metrics(
        test_lA, te_muA_o, te_sigA_o,   test_vA, 'Main dual-head (raw)')
    res['Single-head (cal)']      = full_metrics(
        test_lA, sh_muA_o, sh_sigA_cal, test_vA, 'Single-head (cal)')
    res['Detached Head B (cal)']  = full_metrics(
        test_lA, dt_muA_o, dt_sigA_cal, test_vA, 'Detached Head B (cal)')
    res['HAR-RV']                 = full_metrics(
        test_lA, har_test_lA_pred_o, np.full(len(test_lA), har_sigma_o),
        test_vA, 'HAR-RV')
    res['Constant']               = full_metrics(
        test_lA, const_mu, const_sig, test_vA, 'Constant')
    print("=" * 112)

    # ---------- Auxiliary head evaluation ----------
    print(f"\nAuxiliary head (H_B={H_B}) test metrics:")
    aux_res = full_metrics(test_lB, te_muB_o, te_sigB_cal, test_vB,
                           f'Head B (H={H_B}, cal)')

    # ---------- Delta vs Single-head (the encoder-separation test) ----------
    print("\n" + "-" * 70)
    print("ENCODER-SEPARATION TEST (Thm 10.8) — delta vs Single-head")
    print("-" * 70)
    base_crps  = res['Single-head (cal)']['crps']
    base_qlike = res['Single-head (cal)']['qlike']
    for name in ['Main dual-head (cal)', 'HAR-RV',
                 'Detached Head B (cal)', 'Constant']:
        dc = res[name]['crps']  - base_crps
        dq = res[name]['qlike'] - base_qlike
        rel = 100.0 * dc / base_crps if base_crps > 0 else 0
        verdict = ''
        if name == 'Main dual-head (cal)':
            if rel < -2.0:   verdict = '  ✓ PASS — dual-head beats single by ≥2%'
            elif rel < -0.3: verdict = '  ~ marginal gain (<2%)'
            else:            verdict = '  ✗ FAIL — no encoder separation'
        print(f"  {name:<28} ΔCRPS={dc:+.4f} ({rel:+.2f}%)  ΔQLIKE={dq:+.4f}{verdict}")

    # ---------- Abstention diagnostic (σ̂ separates easy/hard) ----------
    print("\nAbstention diagnostic on σ̂_A (primary head):")
    tau_sig = float(np.quantile(te_sigA_cal, 0.80))
    act = te_sigA_cal <= tau_sig
    print(f"  Coverage at τ_σ (q80) = {act.mean():.3f}")
    if act.sum() > 10:
        c1 = full_metrics(test_lA[act],  te_muA_o[act],
                          te_sigA_cal[act], test_vA[act], 'Confident')
        c2 = full_metrics(test_lA[~act], te_muA_o[~act],
                          te_sigA_cal[~act], test_vA[~act], 'Uncertain')
        if c1['crps'] < c2['crps']:
            print(f"  ✓ σ̂_A separates easy from hard windows")

    # ---------- Save ----------
    torch.save({
        'state_dict': model.state_dict(),
        'config': {
            'n_features': len(FEATURE_COLS),
            'hidden': HIDDEN, 'num_layers': NUM_LAYERS,
            'dropout': DROPOUT, 'c_sigma': C_SIGMA,
            'T': T, 'H_A': H_A, 'H_B': H_B,
            'lambda_B': LAMBDA_B, 'ema_decay': EMA_DECAY,
        },
        'feature_cols': FEATURE_COLS,
        'mu_f': mu_f, 'sd_f': sd_f, 'eps': EPS,
        'mu_lA': mu_lA, 'sd_lA': sd_lA,
        'mu_lB': mu_lB, 'sd_lB': sd_lB,
        'tau_a': t_A, 'tau_b': t_B,
        'har_coef': har.coef_, 'har_intercept': har.intercept_,
        'har_sigma_orig': har_sigma_o,
        'history': hist_main,
        'test_metrics': {k: {kk: float(vv) for kk, vv in v.items()}
                         for k, v in res.items()},
        'aux_metrics': {k: float(v) for k, v in aux_res.items()},
    }, CKPT_PATH)
    print(f"\nSaved: {CKPT_PATH}")

    # ---------- Plots ----------
    print("\nPlotting...")
    fig = plt.figure(figsize=(20, 12))
    gs  = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)

    # Loss curves
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(hist_main['train'], label='total train')
    ax.plot(hist_main['val'],   label='total val')
    ax.plot(hist_main['L_A'],   label=f'L_A (H={H_A})', alpha=0.6)
    ax.plot(hist_main['L_B'],   label=f'L_B (H={H_B})', alpha=0.6)
    ax.set(xlabel='epoch', ylabel='loss', title='Loss curves')
    ax.legend(); ax.grid(alpha=0.3)

    # Predicted vs realized (primary)
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(te_muA_o, test_lA, s=2, alpha=0.2)
    lo = min(te_muA_o.min(), test_lA.min())
    hi = max(te_muA_o.max(), test_lA.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
    ax.set(xlabel=f'μ̂_A (pred log-vol, H={H_A})',
           ylabel=f'ℓ^(H={H_A}) realized',
           title=f'Primary head scatter (ρ={res["Main dual-head (cal)"]["pearson"]:.3f})')
    ax.grid(alpha=0.3)

    # Coverage: raw vs calibrated
    ax = fig.add_subplot(gs[0, 2])
    levels = np.array([0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95])
    covs_raw = [gauss_coverage(test_lA, te_muA_o, te_sigA_o,   lv) for lv in levels]
    covs_cal = [gauss_coverage(test_lA, te_muA_o, te_sigA_cal, lv) for lv in levels]
    ax.plot(levels, covs_raw, 'o--', color='tab:red',  label='Raw σ̂')
    ax.plot(levels, covs_cal, 's-',  color='tab:blue', label='Calibrated σ̂')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Ideal')
    ax.set(xlabel='Nominal coverage', ylabel='Empirical coverage',
           title=f'Reliability curve — Head A (H={H_A})')
    ax.legend(); ax.grid(alpha=0.3)

    # CRPS by σ decile
    ax = fig.add_subplot(gs[1, 0])
    order = np.argsort(te_sigA_cal)
    deciles = np.array_split(order, 10)
    crps_per_decile = []
    for d in deciles:
        c = crps_gaussian_scalar(
            torch.tensor(test_lA[d], dtype=torch.float32),
            torch.tensor(te_muA_o[d], dtype=torch.float32),
            torch.log(torch.tensor(te_sigA_cal[d],
                                   dtype=torch.float32).clamp(min=1e-6))
        ).mean().item()
        crps_per_decile.append(c)
    ax.bar(range(1, 11), crps_per_decile, color='tab:purple')
    ax.set(xlabel='predicted σ̂_A decile (1=confident)', ylabel='CRPS',
           title='CRPS vs σ̂_A decile')
    ax.grid(alpha=0.3)

    # Model comparison table
    ax = fig.add_subplot(gs[1, 1]); ax.axis('off')
    rows = []
    for name in ['Main dual-head (cal)', 'Single-head (cal)',
                 'Detached Head B (cal)', 'HAR-RV', 'Constant']:
        r = res[name]
        rows.append([name, f"{r['crps']:.4f}", f"{r['pearson']:.4f}",
                     f"{r['qlike']:.4f}", f"{r['cov80']:.3f}"])
    cols = ['Model', 'CRPS', 'ρ', 'QLIKE', 'cov80']
    tbl = ax.table(cellText=rows, colLabels=cols,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.1, 1.5)
    ax.set_title(f'Test metrics (H_A={H_A})', pad=10)

    # Per-asset Pearson
    ax = fig.add_subplot(gs[1, 2])
    asset_ids = [w['asset'] for w in test_w]
    pers = []
    for a in sorted(set(asset_ids)):
        m = np.array(asset_ids) == a
        p = float(np.corrcoef(np.exp(te_muA_o[m]), test_vA[m])[0, 1])
        pers.append((a, p))
    pers.sort(key=lambda x: x[1])
    names = [p[0] for p in pers]
    vals  = [p[1] for p in pers]
    colors = ['tab:red' if v < 0.5 else 'tab:green' for v in vals]
    ax.barh(names, vals, color=colors)
    ax.axvline(0.5, ls='--', color='gray')
    ax.set(xlabel=f'Pearson ρ(exp(μ̂_A), v^(H={H_A}))',
           title='Per-asset Pearson correlation')
    ax.grid(alpha=0.3)

    plt.suptitle(
        f'Dual-Head TLSTM v4.0 — {len(assets)} assets, T={T}, '
        f'H_A={H_A} (primary), H_B={H_B} (aux), λ_B={LAMBDA_B}',
        fontsize=13, y=0.995)
    plt.savefig('dual_head_tlstm_v4.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("Saved: dual_head_tlstm_v4.png")


if __name__ == "__main__":
    main()