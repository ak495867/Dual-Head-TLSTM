"""
Dual-Head TLSTM v2.0 — spec-compliant implementation.
======================================================
Implements the mathematical specification exactly:

  §2.4    Calendar-date split with embargo of T+H            (Def 2.8, 2.9)
  §2.5    Per-asset windowing                                (Prop 2.13, Spec 2.14)
  §3      Causal 12-dimensional feature vector               (Def 3.1–3.5, Prop 3.6)
  §4.1    Train-fold affine standardization                  (Def 4.1, Lemma 4.2)
  §4.2    Train-fold vol scale                               (Def 4.4, Lemma 4.5)
  §6      Shared LSTM encoder, terminal hidden state         (Def 6.1, 6.2)
  §7      Head A: (μ, log σ) per horizon, clamp ±c_σ         (Def 7.1–7.2)
  §8      Head B: log-realized-vol forecast                  (Def 8.1–8.2)
  §9.2    Head-A Gaussian CRPS                               (Def 9.5)
  §9.3    Head-B MSE in log space                            (Def 9.8)
  §9.3    Consistency loss with stop-gradient                (Def 9.9)
  §9.4    Composite L_total = L_A + λ_B L_B + λ_C L_C        (Def 9.4)
  §12     Volatility-thresholded abstention                  (Def 12.1–12.2)
  §14     AdamW + clipping + cosine + early stop             (§14.1–14.6)
  §15.2   Non-overlapping annualized Sharpe                  (Def 15.2)
  §15.4   Mandatory baselines                                (4 ablations)

Saves: dual_head_tlstm_v2.pth
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

T         = 30          # §5  window length
H         = 5           # §1.3 horizon
HIDDEN    = 64          # §6.1 hidden width
NUM_LAYERS= 1
DROPOUT   = 0.1
C_SIGMA   = 6.0         # §7.2 log-σ clamp constant

BATCH_SIZE= 512
EPOCHS    = 100
LR        = 5e-4        # §14.1 η_0
GAMMA     = 1e-5        # §14.1 decoupled weight decay
LAMBDA_B  = 0.3         # §9.4
LAMBDA_C  = 0.01         # §9.4
PATIENCE  = 15          # §14.4
KAPPA_0   = 0.80        # §12.1 target coverage

CKPT_PATH = 'dual_head_tlstm_v2.pth'
CACHE_DIR = './cache_v2'
os.makedirs(CACHE_DIR, exist_ok=True)

# §3.3 — the 12-dimensional feature vector, in order
FEATURE_COLS = [
    'r1','r5','r20',
    'vol20','vol60',
    'ma5_ratio','ma20_ratio',
    'rsi','vol_pct',
    'vix','vix_pct','tnx_pct',
]

# ==================================================================
# §3  Data download and features
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
    if 'Close' not in df.columns or 'Volume' not in df.columns: return None
    df = df[['Close','Volume']].dropna()
    try: df.to_parquet(path)
    except Exception: pass
    return df


def download_macro(start, end):
    path = os.path.join(CACHE_DIR, f"macro_{start}_{end}.parquet")
    if os.path.exists(path):
        try: return pd.read_parquet(path)
        except Exception: pass
    out = {}
    for sym, name in [('^VIX','vix'), ('^TNX','tnx')]:
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
    """§3.1–3.5. Every coordinate is F_t-measurable (Prop 3.6)."""
    close  = df['Close'].astype(float)
    volume = df['Volume'].astype(float)
    logc   = np.log(close)
    f = pd.DataFrame(index=df.index)

    # Def 3.1 — log returns over k days
    f['r1']  = logc.diff(1)
    f['r5']  = logc.diff(5)
    f['r20'] = logc.diff(20)

    # Def 3.2 — realized volatility
    f['vol20'] = f['r1'].rolling(20).std()
    f['vol60'] = f['r1'].rolling(60).std()

    # Def 3.3 — MA ratios
    f['ma5_ratio']  = close / close.rolling(5).mean()  - 1.0
    f['ma20_ratio'] = close / close.rolling(20).mean() - 1.0

    # Def 3.4 — Wilder RSI, n=14
    delta = close.diff()
    U = delta.clip(lower=0)
    D = (-delta).clip(lower=0)
    n = 14
    Ubar = U.ewm(alpha=1/n, adjust=False).mean()
    Dbar = D.ewm(alpha=1/n, adjust=False).mean()
    f['rsi'] = 100.0 * (1.0 - 1.0 / (1.0 + Ubar / (Dbar + 1e-9)))

    # Def 3.5 — relative volume
    f['vol_pct'] = volume / (volume.rolling(20).mean() + 1e-9) - 1.0

    # §3.2 — macro features (forward-fill only, Prop 2.7)
    macro_ff = macro.reindex(f.index, method='ffill')
    f['vix']     = macro_ff['vix']
    f['vix_pct'] = macro_ff['vix'].pct_change(1)
    f['tnx_pct'] = macro_ff['tnx'].pct_change(1)

    # 1-day log-return for target construction; dropped from feature list
    f['log_ret'] = f['r1']

    f = f.replace([np.inf, -np.inf], np.nan).dropna()
    return f


# ==================================================================
# §2.5  Per-asset windowing with date tracking
# ==================================================================
def build_windows_for_asset(feat, feature_cols, T, H, asset_id):
    """
    Returns list of tuples (X, y_ret, y_vol, date, asset_id, i) with:
      X      : (T, d) features from dates i-T+1..i       (F_t-measurable)
      y_ret  : (H,)   log returns from i+1..i+H          (F_{t+H}-measurable)
      y_vol  : scalar sqrt(mean(y_ret^2))                (Def 1.3)
      date   : end date of window (forecast origin)
    """
    X_all    = feat[feature_cols].values
    log_ret  = feat['log_ret'].values
    dates    = feat.index
    n        = len(feat)
    out = []
    for i in range(T - 1, n - H):
        Xw    = X_all[i - T + 1 : i + 1].astype(np.float32)
        y_ret = log_ret[i + 1 : i + H + 1].astype(np.float32)
        y_vol = float(np.sqrt(np.mean(y_ret * y_ret)))
        out.append((Xw, y_ret, y_vol, dates[i], asset_id, i))
    return out


# ==================================================================
# §6–8  Model
# ==================================================================
class DualHeadTLSTM(nn.Module):
    """
    §6.1  LSTM encoder, terminal hidden state h_T.
    §7.2  Head A: (μ_1..μ_H, log σ_1..log σ_H) with clamp at ±c_σ.
    §8.2  Head B: log realized volatility scalar.
    """
    def __init__(self, n_features, hidden=64, num_layers=1,
                 dropout=0.1, horizon=5, c_sigma=6.0):
        super().__init__()
        self.horizon = horizon
        self.c_sigma = c_sigma
        self.lstm = nn.LSTM(n_features, hidden, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head_a = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2 * horizon))
        self.head_b = nn.Sequential(
            nn.Linear(hidden, 32), nn.GELU(),
            nn.Linear(32, 1))

    def forward(self, x, detach_b=False):
        out, _ = self.lstm(x)
        h = self.drop(out[:, -1, :])                       # §6.2 terminal state

        a = self.head_a(h)
        mu        = a[:, :self.horizon]                    # §7.2
        log_sigma = torch.clamp(a[:, self.horizon:], -self.c_sigma, self.c_sigma)

        h_b = h.detach() if detach_b else h                # §15.4 baseline 3
        log_v = self.head_b(h_b).squeeze(-1)               # §8.2

        return mu, log_sigma, log_v


# ==================================================================
# §9.1–9.2  CRPS for Gaussian (Def 9.2)
# ==================================================================
def crps_gaussian(y, mu, log_sigma):
    """Gaussian CRPS. y, mu, log_sigma: (B, H). Returns (B, H)."""
    sigma = torch.exp(log_sigma).clamp(min=1e-6)
    z = (y - mu) / sigma
    phi = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


# ==================================================================
# §9.4  Composite loss
# ==================================================================
def composite_loss(mu, log_sigma, log_v, y_ret, y_vol, lam_B, lam_C):
    """
    L_total = L_A + λ_B L_B + λ_C L_C   (Def 9.4)

      L_A : Head-A CRPS over H horizons       (Def 9.5)
      L_B : MSE in log-vol space              (Def 9.8)
      L_C : consistency with stop-gradient    (Def 9.9)
    """
    # §9.2 — Head A
    L_A = crps_gaussian(y_ret, mu, log_sigma).mean()

    # §9.3 — Head B, log-space MSE
    log_v_target = torch.log(y_vol.clamp(min=1e-6))
    L_B = F.mse_loss(log_v, log_v_target)

    # §9.3 — consistency, with sg[ℓ̂^vol]
    if lam_C > 0:
        sigma     = torch.exp(log_sigma)                    # (B, H)
        sigma_bar = torch.sqrt((sigma * sigma).mean(dim=1)) # (B,)
        L_C = F.mse_loss(torch.log(sigma_bar + 1e-6), log_v.detach())
    else:
        L_C = torch.tensor(0.0, device=mu.device)

    total = L_A + lam_B * L_B + lam_C * L_C
    return total, float(L_A), float(L_B), float(L_C)


# ==================================================================
# §14  Training
# ==================================================================
def train_dual_head(model, train_loader, val_loader, lam_B, lam_C,
                    epochs, lr, gamma, patience, detach_b=False):
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=gamma)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val   = float('inf')
    best_state = None
    wait       = 0
    history    = {'train': [], 'val': []}

    for ep in range(epochs):
        # ---- train ---------------------------------------------------
        model.train()
        tot, n = 0.0, 0
        for xb, yrb, yvb in train_loader:
            xb, yrb, yvb = xb.to(DEVICE), yrb.to(DEVICE), yvb.to(DEVICE)
            mu, ls, lv = model(xb, detach_b=detach_b)
            loss, _, _, _ = composite_loss(mu, ls, lv, yrb, yvb, lam_B, lam_C)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # Def 14.3
            opt.step()
            tot += loss.item() * len(xb); n += len(xb)
        train_loss = tot / n
        history['train'].append(train_loss)

        # ---- val -----------------------------------------------------
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for xb, yrb, yvb in val_loader:
                xb, yrb, yvb = xb.to(DEVICE), yrb.to(DEVICE), yvb.to(DEVICE)
                mu, ls, lv = model(xb, detach_b=detach_b)
                loss, _, _, _ = composite_loss(mu, ls, lv, yrb, yvb, lam_B, lam_C)
                tot += loss.item() * len(xb); n += len(xb)
        val_loss = tot / n
        history['val'].append(val_loss)

        sched.step()

        improved = val_loss < best_val - 1e-5
        if improved:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait       = 0
        else:
            wait += 1
        if ep % 5 == 0 or improved:
            print(f"  ep {ep+1:3d} | train={train_loss:.4f}  val={val_loss:.4f}"
                  f"{' *' if improved else ''}")
        if wait >= patience:
            print(f"  early stop at ep {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_val


# ==================================================================
# §15  Prediction helpers
# ==================================================================
@torch.no_grad()
def predict(model, X, detach_b=False, bs=2048):
    """Returns (mu, sigma, v̂) all in rescaled units."""
    model.eval()
    Xt = torch.as_tensor(X, dtype=torch.float32)
    mus, sigs, vs = [], [], []
    for i in range(0, len(Xt), bs):
        xb = Xt[i:i+bs].to(DEVICE)
        mu, ls, lv = model(xb, detach_b=detach_b)
        mus.append(mu.cpu().numpy())
        sigs.append(torch.exp(ls).cpu().numpy())
        vs.append(torch.exp(lv).cpu().numpy())
    return (np.concatenate(mus, 0),
            np.concatenate(sigs, 0),
            np.concatenate(vs, 0))


# ==================================================================
# §15.1  Metrics
# ==================================================================
def directional_accuracy(mu, y_ret):
    return float((np.sign(mu[:, 0]) == np.sign(y_ret[:, 0])).mean())


def volatility_correlation(v, y_vol):
    if len(v) < 2: return float('nan')
    return float(np.corrcoef(v, y_vol)[0, 1])


def coverage_and_selective_accuracy(mu, v, y_ret, tau_vol):
    accepted = v <= tau_vol
    kappa = float(accepted.mean())
    if accepted.sum() < 5:
        return kappa, float('nan')
    sa = float((np.sign(mu[accepted, 0]) == np.sign(y_ret[accepted, 0])).mean())
    return kappa, sa


# §15.2 — non-overlapping annualized Sharpe
def non_overlapping_sharpe(mu, v, y_ret, asset_ids, H):
    df = pd.DataFrame({
        'asset': asset_ids,
        'mu1':   mu[:, 0],
        'v':     v,
        'r1':    y_ret[:, 0],
    })
    sharpes = []
    for a, g in df.groupby('asset'):
        g = g.reset_index(drop=True).iloc[::H]         # every H-th sample
        if len(g) < 5: continue
        pi  = np.tanh(g['mu1'].values / (g['v'].values + 1e-6))
        pnl = pi * g['r1'].values
        s   = pnl.std()
        if s > 1e-8:
            sharpes.append(np.sqrt(252.0 / H) * pnl.mean() / s)
    return float(np.mean(sharpes)) if sharpes else float('nan')


# ==================================================================
# Main pipeline
# ==================================================================
def main():
    print(f"\nConfig: T={T}, H={H}, hidden={HIDDEN}, "
          f"λ_B={LAMBDA_B}, λ_C={LAMBDA_C}, κ_0={KAPPA_0}")

    # ---------- [1] Data -------------------------------------------
    print("\n[1/8] Downloading data...")
    macro = download_macro(START_DATE, END_DATE)
    train_assets = {}
    for tk in TRAIN_TICKERS:
        raw = download_prices(tk, START_DATE, END_DATE)
        if raw is None: continue
        feat = compute_features(raw, macro)
        if len(feat) < T + H + 100: continue
        train_assets[tk] = feat
        print(f"  {tk}: {len(feat)} rows")
    print(f"  Loaded {len(train_assets)} assets")

    # ---------- [2] Windows ----------------------------------------
    print("\n[2/8] Building per-asset windows...")
    all_windows = []
    for tk, feat in train_assets.items():
        all_windows.extend(build_windows_for_asset(feat, FEATURE_COLS, T, H, tk))
    print(f"  Total windows: {len(all_windows)}")

    # ---------- [3] Calendar-date split (§2.4, Def 2.8–2.9) --------
    print("\n[3/8] Calendar-date split with embargo T+H...")
    master = sorted(train_assets['SPY'].index)
    m = len(master)
    t1_idx = int(0.60 * m)
    t2_idx = int(0.80 * m)
    emb = T + H

    train_end_date  = master[t1_idx - emb - 1]
    val_start_date  = master[t1_idx]
    val_end_date    = master[t2_idx - emb - 1]
    test_start_date = master[t2_idx]

    print(f"  Train ends at   : {train_end_date.date()}")
    print(f"  Val  in         : [{val_start_date.date()}, {val_end_date.date()}]")
    print(f"  Test starts at  : {test_start_date.date()}")

    train_set = [w for w in all_windows if w[3] <= train_end_date]
    val_set   = [w for w in all_windows if val_start_date <= w[3] <= val_end_date]
    test_set  = [w for w in all_windows if w[3] >= test_start_date]
    print(f"  Windows: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    def stack(ws):
        X  = np.stack([w[0] for w in ws]).astype(np.float32)
        yr = np.stack([w[1] for w in ws]).astype(np.float32)
        yv = np.array([w[2] for w in ws], dtype=np.float32)
        dt = np.array([w[3] for w in ws])
        a  = np.array([w[4] for w in ws])
        return X, yr, yv, dt, a

    train_X, train_yr, train_yv, train_dt, train_ast = stack(train_set)
    val_X,   val_yr,   val_yv,   val_dt,   val_ast   = stack(val_set)
    test_X,  test_yr,  test_yv,  test_dt,  test_ast  = stack(test_set)

    # ---------- [4] Train-fold standardization (§4.1, §4.2) --------
    print("\n[4/8] Fitting train-fold scaler and vol scale...")
    mu_f = train_X.mean(axis=(0, 1)).astype(np.float32)
    sd_f = (train_X.std(axis=(0, 1)) + 1e-8).astype(np.float32)
    train_X = (train_X - mu_f) / sd_f
    val_X   = (val_X   - mu_f) / sd_f
    test_X  = (test_X  - mu_f) / sd_f

    s_vol = float(train_yv.mean())
    train_yr /= s_vol; train_yv /= s_vol
    val_yr   /= s_vol; val_yv   /= s_vol
    test_yr  /= s_vol; test_yv  /= s_vol

    print(f"  Feature scale: mu range [{mu_f.min():.3f},{mu_f.max():.3f}], "
          f"sd mean {sd_f.mean():.4f}")
    print(f"  Vol scale s_vol = {s_vol:.5f}")

    # ---------- Loaders --------------------------------------------
    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_X), torch.tensor(train_yr),
                      torch.tensor(train_yv)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(val_X), torch.tensor(val_yr),
                      torch.tensor(val_yv)),
        batch_size=BATCH_SIZE, shuffle=False)

    # ---------- [5] Train main model -------------------------------
    print("\n[5/8] Training main dual-head model...")
    model = DualHeadTLSTM(len(FEATURE_COLS), HIDDEN, NUM_LAYERS,
                          DROPOUT, H, C_SIGMA).to(DEVICE)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    t0 = time.time()
    hist_main, best_val = train_dual_head(
        model, train_loader, val_loader,
        LAMBDA_B, LAMBDA_C, EPOCHS, LR, GAMMA, PATIENCE,
        detach_b=False)
    print(f"  Time: {(time.time()-t0)/60:.1f} min | best val loss = {best_val:.4f}")

    # ---------- [6] Val threshold and test predictions -------------
    print("\n[6/8] Computing τ_vol and evaluating on test...")
    val_mu, val_sig, val_v = predict(model, val_X)
    tau_vol = float(np.quantile(val_v, KAPPA_0))              # §12.1
    print(f"  τ_vol = Q_{KAPPA_0:.2f}(v̂_val) = {tau_vol:.4f}  "
          f"(in original units: {tau_vol*s_vol:.5f})")

    # Test predictions
    test_mu, test_sig, test_v = predict(model, test_X)

    # Unscale for reporting
    test_mu_o = test_mu * s_vol
    test_v_o  = test_v  * s_vol
    test_yr_o = test_yr * s_vol
    test_yv_o = test_yv * s_vol
    tau_o     = tau_vol * s_vol

    # Metrics
    dir_acc = directional_accuracy(test_mu, test_yr)
    vcorr   = volatility_correlation(test_v, test_yv)
    cov, sa = coverage_and_selective_accuracy(test_mu, test_v, test_yr, tau_vol)
    sharpe  = non_overlapping_sharpe(test_mu, test_v, test_yr, test_ast, H)

    print(f"  Main model on test:")
    print(f"    Directional accuracy : {dir_acc:.4f}")
    print(f"    Volatility corr      : {vcorr:.4f}")
    print(f"    Coverage (κ_0={KAPPA_0})     : {cov:.4f}")
    print(f"    Selective accuracy   : {sa:.4f}")
    print(f"    Non-overlap Sharpe   : {sharpe:.3f}")

    # ---------- [7] Baselines (§15.4) ------------------------------
    print("\n[7/8] Baselines...")

    # B2: single-head (λ_B = λ_C = 0)
    print("  B2: single-head (λ_B=λ_C=0)...")
    torch.manual_seed(SEED); np.random.seed(SEED)
    m_sh = DualHeadTLSTM(len(FEATURE_COLS), HIDDEN, NUM_LAYERS,
                         DROPOUT, H, C_SIGMA).to(DEVICE)
    train_dual_head(m_sh, train_loader, val_loader,
                    0.0, 0.0, EPOCHS, LR, GAMMA, PATIENCE, detach_b=False)
    sh_mu, _, sh_v = predict(m_sh, test_X)
    sh_val_v = predict(m_sh, val_X)[2]
    tau_sh = float(np.quantile(sh_val_v, KAPPA_0))
    sh_dir = directional_accuracy(sh_mu, test_yr)
    sh_vc  = volatility_correlation(sh_v, test_yv)
    sh_c, sh_sa = coverage_and_selective_accuracy(sh_mu, sh_v, test_yr, tau_sh)
    sh_sr  = non_overlapping_sharpe(sh_mu, sh_v, test_yr, test_ast, H)

    # B3: detached Head B
    print("  B3: detached Head B...")
    torch.manual_seed(SEED); np.random.seed(SEED)
    m_dt = DualHeadTLSTM(len(FEATURE_COLS), HIDDEN, NUM_LAYERS,
                         DROPOUT, H, C_SIGMA).to(DEVICE)
    train_dual_head(m_dt, train_loader, val_loader,
                    LAMBDA_B, LAMBDA_C, EPOCHS, LR, GAMMA, PATIENCE,
                    detach_b=True)
    dt_mu, _, dt_v = predict(m_dt, test_X, detach_b=True)
    dt_val_v = predict(m_dt, val_X, detach_b=True)[2]
    tau_dt = float(np.quantile(dt_val_v, KAPPA_0))
    dt_dir = directional_accuracy(dt_mu, test_yr)
    dt_vc  = volatility_correlation(dt_v, test_yv)
    dt_c, dt_sa = coverage_and_selective_accuracy(dt_mu, dt_v, test_yr, tau_dt)
    dt_sr  = non_overlapping_sharpe(dt_mu, dt_v, test_yr, test_ast, H)

    # B1: constant predictor (majority direction of train)
    maj = np.sign(train_yr.mean(axis=0)[0])
    const_acc = float((np.full(len(test_yr), maj) == np.sign(test_yr[:, 0])).mean())

    # B4: no-abstention (main model, τ = ∞ ⇒ coverage 1)
    noab_c, noab_sa = 1.0, dir_acc   # all samples accepted, same acc

    # ---------- Summary table --------------------------------------
    print("\n" + "=" * 78)
    print(f"{'Model':<34} {'DirAcc':>8} {'VolCorr':>8} "
          f"{'Cov':>7} {'SelAcc':>8} {'Sharpe':>8}")
    print("-" * 78)
    rows = [
        ('Main dual-head',         dir_acc, vcorr, cov,     sa,      sharpe),
        ('Single-head (λ_B=λ_C=0)',sh_dir,  sh_vc, sh_c,    sh_sa,   sh_sr),
        ('Detached Head B',        dt_dir,  dt_vc, dt_c,    dt_sa,   dt_sr),
        ('No abstention (τ=∞)',    dir_acc, vcorr, noab_c,  noab_sa, sharpe),
    ]
    for name, a, v, c, s, sr in rows:
        print(f"{name:<34} {a:>8.4f} {v:>8.4f} {c:>7.3f} "
              f"{s:>8.4f} {sr:>8.3f}")
    print(f"{'Constant predictor':<34} {const_acc:>8.4f} "
          f"{'—':>8} {'—':>7} {'—':>8} {'—':>8}")
    print("=" * 78)

    # ---------- [8] Save model + plots -----------------------------
    torch.save({
        'state_dict': model.state_dict(),
        'config': {
            'n_features': len(FEATURE_COLS),
            'hidden': HIDDEN, 'num_layers': NUM_LAYERS,
            'dropout': DROPOUT, 'horizon': H,
            'window': T, 'c_sigma': C_SIGMA,
        },
        'feature_cols': FEATURE_COLS,
        'mu_f': mu_f, 'sd_f': sd_f,
        's_vol': s_vol,
        'tau_vol': tau_vol, 'kappa_0': KAPPA_0,
        'history': hist_main,
        'test_metrics': {
            'dir_acc': dir_acc, 'vol_corr': vcorr,
            'coverage': cov, 'sel_acc': sa, 'sharpe': sharpe,
        },
    }, CKPT_PATH)
    print(f"\nSaved checkpoint: {CKPT_PATH}")

    # Plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    ax.plot(hist_main['train'], label='train')
    ax.plot(hist_main['val'],   label='val')
    ax.set(xlabel='epoch', ylabel='loss', title='Loss curves (main)')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(test_v_o, test_yv_o, s=2, alpha=0.3)
    lim = [min(test_v_o.min(), test_yv_o.min()),
           max(test_v_o.max(), test_yv_o.max())]
    ax.plot(lim, lim, 'r--', lw=1)
    ax.set(xlabel='v̂ (pred vol)', ylabel='realized vol',
           title=f'Vol scatter (ρ={vcorr:.3f})')
    ax.grid(alpha=0.3)

    # Directional accuracy vs vol decile
    ax = axes[0, 2]
    v_sorted = np.argsort(test_v)
    deciles  = np.array_split(v_sorted, 10)
    dec_acc  = []
    for d in deciles:
        a = (np.sign(test_mu[d, 0]) == np.sign(test_yr[d, 0])).mean()
        dec_acc.append(a)
    ax.bar(range(1, 11), dec_acc)
    ax.axhline(0.5, ls='--', color='gray')
    ax.set(xlabel='vol decile (1=low)', ylabel='direction accuracy',
           title='Direction accuracy vs v̂ decile')
    ax.grid(alpha=0.3)

    # Coverage vs selective accuracy
    ax = axes[1, 0]
    kappas  = np.linspace(0.05, 0.95, 19)
    coords  = []
    for k0 in kappas:
        t = float(np.quantile(val_v, k0))
        c, s = coverage_and_selective_accuracy(test_mu, test_v, test_yr, t)
        coords.append((c, s))
    coords = np.array(coords)
    ax.plot(coords[:, 0], coords[:, 1], 'o-')
    ax.axhline(0.5, ls='--', color='gray')
    ax.set(xlabel='coverage κ', ylabel='selective accuracy',
           title='Coverage vs selective accuracy')
    ax.grid(alpha=0.3)

    # Sharpe per asset
    ax = axes[1, 1]
    sr_per_asset = {}
    for a in np.unique(test_ast):
        m = test_ast == a
        s = non_overlapping_sharpe(test_mu[m], test_v[m], test_yr[m],
                                   test_ast[m], H)
        sr_per_asset[a] = s
    tks = list(sr_per_asset.keys())
    vals = [sr_per_asset[t] for t in tks]
    colors = ['tab:green' if v > 0 else 'tab:red' for v in vals]
    ax.barh(tks, vals, color=colors)
    ax.axvline(0, ls='--', color='gray')
    ax.set(xlabel='Sharpe (non-overlap)', title='Test Sharpe per asset')
    ax.grid(alpha=0.3)

    # Summary text
    ax = axes[1, 2]; ax.axis('off')
    lines = [
        f"Train windows : {len(train_X):,}",
        f"Val windows   : {len(val_X):,}",
        f"Test windows  : {len(test_X):,}",
        f"Params        : {sum(p.numel() for p in model.parameters()):,}",
        "",
        f"Directional accuracy : {dir_acc:.4f}",
        f"Volatility corr      : {vcorr:.4f}",
        f"Coverage κ           : {cov:.4f}",
        f"Selective accuracy   : {sa:.4f}",
        f"Sharpe (non-overlap) : {sharpe:.3f}",
        "",
        f"τ_vol (orig)         : {tau_o:.5f}",
    ]
    for i, line in enumerate(lines):
        ax.text(0.02, 0.98 - i * 0.07, line, fontsize=10,
                family='monospace', transform=ax.transAxes, va='top')

    plt.suptitle(f'Dual-Head TLSTM v2.0 — trained on {len(train_assets)} assets, '
                 f'chronological split (T={T}, H={H})', fontsize=13)
    plt.tight_layout()
    plt.savefig('dual_head_tlstm_v2.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("Saved: dual_head_tlstm_v2.png")


if __name__ == "__main__":
    main()