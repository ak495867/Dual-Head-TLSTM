"""
Dual Head TLSTM V1
==================
  - Dual Head TLSTM spec: formal causality, per-asset windows, embargoed splits,
    auxiliary head, selective prediction, distributional head (μ, σ), CRPS loss, multi-horizon

Saves: dual_head_tlstm_v1.pth
"""

import os, time, math, warnings
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
# Config
# ==================================================================
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

TRAIN_TICKERS = [
    'SPY','QQQ','IWM','DIA','MDY','RSP','VTI',
    'XLE','XLF','XLK','XLV','XLI','XLP','XLY','XLU','XLB',
    'AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA','JPM',
    'JNJ','WMT','XOM','BAC','V','UNH','PG','HD','CVX','ABBV',
    'KO','PEP','ORCL','CSCO'
]

TEST_TICKERS = [
    'GLD','SLV','USO','UNG',
    'TLT','IEF','SHY','HYG','LQD','EMB',
    'EFA','EEM','FXI','EWJ','EWZ',
    'MTUM','VLUE','QUAL','USMV',
    'XBI','XRT','XOP'
]

START_DATE = '2010-01-01'
END_DATE   = '2023-12-31'

WINDOW     = 30
HORIZON    = 5
HIDDEN     = 64
NUM_LAYERS = 1
DROPOUT    = 0.1
BATCH_SIZE = 512
EPOCHS     = 100
LR         = 5e-4
GAMMA      = 1e-5
LAMBDA_VOL = 1.0        # weight on Head B
PATIENCE   = 15
EMBARGO    = WINDOW + HORIZON
CACHE_DIR  = './cache_v1'
CKPT_PATH  = 'dual_head_tlstm_v1.pth'
os.makedirs(CACHE_DIR, exist_ok=True)

FEATURE_COLS = [
    'r1','r5','r20',
    'vol20','vol60',
    'ma5_ratio','ma20_ratio',
    'rsi','vol_pct',
    'vix','vix_pct','tnx_pct',
]

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
    close  = df['Close'].astype(float)
    volume = df['Volume'].astype(float)
    f = pd.DataFrame(index=df.index)

    logc = np.log(close)
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
    Ub = U.ewm(alpha=1/n, adjust=False).mean()
    Db = D.ewm(alpha=1/n, adjust=False).mean()
    f['rsi'] = 100 * (1 - 1 / (1 + Ub / (Db + 1e-9)))

    f['vol_pct'] = volume / (volume.rolling(20).mean() + 1e-9) - 1.0

    macro_ff = macro.reindex(f.index, method='ffill')
    f['vix']     = macro_ff['vix']
    f['vix_pct'] = macro_ff['vix'].pct_change(1)
    f['tnx_pct'] = macro_ff['tnx'].pct_change(1)

    f['log_ret'] = f['r1']   # store for target construction

    f = f.replace([np.inf, -np.inf], np.nan).dropna()
    return f


def make_windows_per_asset(feat, window, horizon):
    """
    Build per-asset (X, y_ret, y_vol) triples.
      X_i    : features[i-window+1 : i+1]     -> (window, d)
      y_ret_i: log_ret[i+1 : i+horizon+1]     -> (horizon,)
      y_vol_i: sqrt(mean(y_ret_i^2))          -> (1,)
    Feature at row i uses info up to close of day i (causal).
    Target uses days i+1 ... i+horizon (future).
    """
    X = feat[FEATURE_COLS].values
    r = feat['log_ret'].values
    n = len(feat)
    Xw, yw_ret, yw_vol = [], [], []
    for i in range(window - 1, n - horizon):
        Xw.append(X[i - window + 1 : i + 1])
        y  = r[i + 1 : i + horizon + 1]
        yw_ret.append(y)
        yw_vol.append(math.sqrt(np.mean(y * y)))
    return (np.asarray(Xw, dtype=np.float32),
            np.asarray(yw_ret, dtype=np.float32),
            np.asarray(yw_vol, dtype=np.float32).reshape(-1, 1))


# ==================================================================
# Model
# ==================================================================
class DualHeadTLSTM(nn.Module):
    def __init__(self, n_features, hidden=64, num_layers=1, dropout=0.1, horizon=5):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(n_features, hidden, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)

        # Head A: (mu_1..mu_H, log_sigma_1..log_sigma_H)
        self.head_a = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2 * horizon))

        # Head B: log of realized vol over next H days
        self.head_b = nn.Sequential(
            nn.Linear(hidden, 32), nn.GELU(),
            nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.drop(out[:, -1, :])          # last hidden state
        a = self.head_a(h)
        mu        = a[:, :self.horizon]
        log_sigma = a[:, self.horizon:]
        log_sigma = torch.clamp(log_sigma, -6.0, 6.0)
        log_v     = self.head_b(h)             # (B, 1)
        return mu, log_sigma, log_v


# ==================================================================
# Losses
# ==================================================================
def crps_gaussian(y, mu, log_sigma):
    """y, mu, log_sigma: (B, H). Returns per-sample mean CRPS over H."""
    sigma = torch.exp(log_sigma) + 1e-6
    z = (y - mu) / sigma
    phi = torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))
    return crps.mean(dim=1)


def composite_loss(mu, log_sigma, log_v, y_ret, y_vol, lam_vol=1.0):
    L_A = crps_gaussian(y_ret, mu, log_sigma).mean()
    log_v_target = torch.log(y_vol + 1e-6).squeeze(-1)
    L_B = F.mse_loss(log_v.squeeze(-1), log_v_target)
    return L_A + lam_vol * L_B, L_A.item(), L_B.item()


# ==================================================================
# Training
# ==================================================================
def train_epoch(model, loader, opt, lam_vol):
    model.train()
    total, n = 0.0, 0
    for xb, yrb, yvb in loader:
        xb, yrb, yvb = xb.to(DEVICE), yrb.to(DEVICE), yvb.to(DEVICE)
        mu, ls, lv = model(xb)
        loss, _, _ = composite_loss(mu, ls, lv, yrb, yvb, lam_vol)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * len(xb); n += len(xb)
    return total / n


@torch.no_grad()
def evaluate(model, loader, lam_vol):
    model.eval()
    tot, n = 0.0, 0
    for xb, yrb, yvb in loader:
        xb, yrb, yvb = xb.to(DEVICE), yrb.to(DEVICE), yvb.to(DEVICE)
        mu, ls, lv = model(xb)
        loss, _, _ = composite_loss(mu, ls, lv, yrb, yvb, lam_vol)
        tot += loss.item() * len(xb); n += len(xb)
    return tot / n


@torch.no_grad()
def predict(model, X, bs=2048):
    model.eval()
    mus, sigs, vs = [], [], []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i+bs]).to(DEVICE)
        mu, ls, lv = model(xb)
        mus.append(mu.cpu().numpy())
        sigs.append(torch.exp(ls).cpu().numpy())
        vs.append(torch.exp(lv).cpu().numpy())
    return (np.concatenate(mus), np.concatenate(sigs), np.concatenate(vs))


# ==================================================================
# Pipeline
# ==================================================================
def main():
    print("\n[1/7] Downloading data...")
    macro = download_macro(START_DATE, END_DATE)
    print(f"  Macro: {macro.shape}")

    print(f"\n[2/7] Downloading {len(TRAIN_TICKERS)} training assets...")
    train_assets = {}
    for tk in TRAIN_TICKERS:
        raw = download_prices(tk, START_DATE, END_DATE)
        if raw is None: continue
        feat = compute_features(raw, macro)
        if len(feat) < 500: continue
        train_assets[tk] = feat
        print(f"  {tk}: {len(feat)} rows")

    # ---------- Build per-asset windows (causal, no cross-ticker) ----
    print(f"\n[3/7] Building windows...")
    all_X, all_yret, all_yvol = [], [], []
    for tk, feat in train_assets.items():
        X, yr, yv = make_windows_per_asset(feat, WINDOW, HORIZON)
        if len(X) < 100: continue
        all_X.append(X); all_yret.append(yr); all_yvol.append(yv)
    X_all    = np.concatenate(all_X, axis=0)
    yret_all = np.concatenate(all_yret, axis=0)
    yvol_all = np.concatenate(all_yvol, axis=0)
    n = len(X_all)
    print(f"  Total windows: {n}")

    # ---------- Train-fold-only standardization (Spec 4.1) -----------
    t1 = int(0.60 * n)
    t2 = int(0.80 * n)
    # Embargo removes boundaries (Def 2.9)
    train_idx = np.arange(0,          t1 - EMBARGO)
    val_idx   = np.arange(t1,         t2 - EMBARGO)
    test_idx  = np.arange(t2,         n)
    print(f"  Folds (embargo={EMBARGO}): "
          f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # Global z-score on features, fit on training fold only
    mu_f = X_all[train_idx].mean(axis=(0,1))
    sd_f = X_all[train_idx].std(axis=(0,1)) + 1e-8

    # Also standardize targets: use next-day vol scale from train fold
    vol_scale = yvol_all[train_idx].mean()

    X_all_z = (X_all - mu_f) / sd_f
    yret_scaled = yret_all / vol_scale
    yvol_scaled = yvol_all / vol_scale

    # ---------- Loaders ---------------------------------------------
    def make_loader(idx, shuffle=False):
        ds = TensorDataset(torch.tensor(X_all_z[idx]),
                           torch.tensor(yret_scaled[idx]),
                           torch.tensor(yvol_scaled[idx]))
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)
    train_loader = make_loader(train_idx, shuffle=True)
    val_loader   = make_loader(val_idx,   shuffle=False)
    test_loader  = make_loader(test_idx,  shuffle=False)

    # ---------- Train -----------------------------------------------
    print(f"\n[4/7] Training Dual Head TLSTM V1...")
    model = DualHeadTLSTM(len(FEATURE_COLS), HIDDEN, NUM_LAYERS, DROPOUT, HORIZON).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=GAMMA)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    best_val, best_state, wait = np.inf, None, 0
    history = {'train': [], 'val': []}
    t0 = time.time()
    for ep in range(EPOCHS):
        tr = train_epoch(model, train_loader, opt, LAMBDA_VOL)
        va = evaluate(model, val_loader, LAMBDA_VOL)
        sched.step()
        history['train'].append(tr); history['val'].append(va)
        marker = ''
        if va < best_val - 1e-5:
            best_val = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0; marker = ' *'
        else:
            wait += 1
        if ep % 5 == 0 or marker:
            print(f"  Epoch {ep+1:3d} | train={tr:.4f} val={va:.4f}{marker}")
        if wait >= PATIENCE:
            print(f"  Early stop at epoch {ep+1}")
            break
    print(f"  Time: {(time.time()-t0)/60:.1f} min, best val={best_val:.4f}")
    model.load_state_dict(best_state)

    # ---------- Validation: abstention threshold --------------------
    p_mu, p_sig, p_v = predict(model, X_all_z[val_idx])
    v_threshold = np.quantile(p_v, 0.80)   # abstain in high-vol 20%
    print(f"\n[5/7] Abstention threshold (q80 of val v̂): {v_threshold*vol_scale:.5f}")

    # ---------- In-sample test-fold evaluation ----------------------
    print(f"\n  In-sample test fold (SPY & friends):")
    p_mu_te, p_sig_te, p_v_te = predict(model, X_all_z[test_idx])
    y_te   = yret_all[test_idx]
    yv_te  = yvol_all[test_idx]
    r1_te  = y_te[:, 0]
    mu1_te = p_mu_te[:, 0] * vol_scale
    v_te   = p_v_te * vol_scale

    def metrics(mu1, v, r1, yv_true):
        pred_sign = np.sign(mu1)
        true_sign = np.sign(r1)
        hit = (pred_sign == true_sign).mean()
        # position size: Kelly-style, bounded
        pos = np.tanh(mu1 / (v + 1e-6))
        pnl = pos * r1
        sharpe = pnl.mean() / (pnl.std() + 1e-8) * np.sqrt(252)
        # abstention
        act = v <= v_threshold
        hit_sel = (pred_sign[act] == true_sign[act]).mean() if act.sum() > 0 else np.nan
        cov = act.mean()
        # vol forecast correlation
        vol_corr = np.corrcoef(v, yv_true)[0, 1]
        return hit, sharpe, cov, hit_sel, vol_corr, pos

    hit, sharpe, cov, hit_sel, vcorr, pos = metrics(mu1_te, v_te, r1_te, yv_te)
    print(f"    n={len(r1_te):5d}  hit={hit:.4f}  sharpe={sharpe:.2f}  "
          f"cov={cov:.3f}  sel_hit={hit_sel:.4f}  vol_corr={vcorr:.3f}")

    # ---------- Zero-shot on unseen assets --------------------------
    print(f"\n[6/7] Zero-shot on {len(TEST_TICKERS)} unseen assets...")
    results = {}
    for tk in TEST_TICKERS:
        raw = download_prices(tk, START_DATE, END_DATE)
        if raw is None: continue
        feat = compute_features(raw, macro)
        if len(feat) < 300: continue
        X, yr, yv = make_windows_per_asset(feat, WINDOW, HORIZON)
        if len(X) < 100: continue
        Xz = (X - mu_f) / sd_f

        p_mu, p_sig, p_v = predict(model, Xz)
        mu1 = p_mu[:, 0] * vol_scale
        v   = p_v       * vol_scale
        r1  = yr[:, 0]
        yv_true = yv.flatten()

        hit, sharpe, cov, hit_sel, vcorr, pos = metrics(mu1, v, r1, yv_true)
        results[tk] = dict(n=len(r1), hit=hit, sharpe=sharpe, cov=cov,
                           hit_sel=hit_sel, vcorr=vcorr, pos=pos, r1=r1, v=v)
        print(f"  {tk:5s} n={len(r1):5d}  hit={hit:.4f}  sharpe={sharpe:6.2f}  "
              f"cov={cov:.3f}  sel_hit={hit_sel:.4f}  vol_corr={vcorr:.3f}")

    # ---------- Save -----------------------------------------------
    print(f"\n[7/7] Saving model...")
    torch.save({
        'state_dict':  best_state,
        'config': {
            'n_features': len(FEATURE_COLS), 'hidden': HIDDEN,
            'num_layers': NUM_LAYERS, 'dropout': DROPOUT, 'horizon': HORIZON,
            'window': WINDOW,
        },
        'feature_cols':   FEATURE_COLS,
        'feature_mu':     mu_f,
        'feature_sd':     sd_f,
        'vol_scale':      vol_scale,
        'v_threshold':    v_threshold,
        'history':        history,
    }, CKPT_PATH)
    print(f"  Saved {CKPT_PATH}")

    # ---------- Plots ----------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    ax.plot(history['train'], label='train'); ax.plot(history['val'], label='val')
    ax.set(xlabel='epoch', ylabel='loss', title='Loss'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    tks = list(results.keys())
    hits = [results[t]['hit'] for t in tks]
    ax.barh(tks, hits, color=['tab:green' if h>0.5 else 'tab:red' for h in hits])
    ax.axvline(0.5, ls='--', color='gray')
    ax.set(xlabel='direction hit rate', title='Zero-shot hit rate')

    ax = axes[0, 2]
    vc = [results[t]['vcorr'] for t in tks]
    ax.barh(tks, vc, color='tab:purple')
    ax.set(xlabel='corr(v̂, realized vol)', title='Volatility forecast quality')

    ax = axes[1, 0]
    sh = [results[t]['sharpe'] for t in tks]
    ax.barh(tks, sh, color=['tab:green' if s>0 else 'tab:red' for s in sh])
    ax.axvline(0, ls='--', color='gray')
    ax.set(xlabel='annualized Sharpe', title='Zero-shot Sharpe')

    ax = axes[1, 1]
    covs = [results[t]['cov'] for t in tks]
    ax.barh(tks, covs, color='tab:orange')
    ax.axvline(0.8, ls='--', color='gray', label='target 80%')
    ax.set(xlabel='coverage', title='Coverage at τ (q80 vol)'); ax.legend()

    ax = axes[1, 2]
    sel = [results[t]['hit_sel'] for t in tks]
    ax.barh(tks, sel, color='tab:blue')
    ax.axvline(0.5, ls='--', color='gray')
    ax.set(xlabel='hit rate | acted', title='Selective hit rate')

    plt.suptitle(f'Dual Head TLSTM V1 — trained on {len(train_assets)} assets, '
                 f'zero-shot on {len(tks)}', fontsize=13)
    plt.tight_layout()
    plt.savefig('dual_head_tlstm_v1.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("  Saved dual_head_tlstm_v1.png")


if __name__ == "__main__":
    main()