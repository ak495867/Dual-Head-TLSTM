"""
Dual-Head TLSTM for Volatility v3.0
====================================
Head A: distributional log total realized vol
Head B: log downside semivol (auxiliary)
Consistency: μ_A ≈ sg[ℓ_B + ½ log 2]

Causality (§2), features (§3), scaler (§4.1) unchanged from v2.0.
Benchmark: HAR-RV (Corsi 2009) — the standard vol baseline.

Saves: dual_head_tlstm_vol_v3.pth
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

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ---------- Config ----------
TRAIN_TICKERS = [
    'SPY','QQQ','IWM','DIA','MDY','RSP','VTI',
    'XLE','XLF','XLK','XLV','XLI','XLP','XLY','XLU','XLB',
    'AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA','JPM',
    'JNJ','WMT','XOM','BAC','V','UNH','PG','HD','CVX','ABBV',
    'KO','PEP','ORCL','CSCO'
]
START_DATE = '2012-01-01'
END_DATE   = '2023-12-31'

T          = 30         # window length
H          = 5          # forecast horizon (trading days)
HIDDEN     = 64
NUM_LAYERS = 1
DROPOUT    = 0.1
C_SIGMA    = 6.0

BATCH_SIZE = 512
EPOCHS     = 150
LR         = 5e-4
GAMMA      = 1e-5
LAMBDA_B   = 1.0
LAMBDA_C   = 0.2
PATIENCE   = 20
KAPPA_0    = 0.80

CKPT_PATH  = 'dual_head_tlstm_vol_v3.pth'
CACHE_DIR  = './cache_v3'
os.makedirs(CACHE_DIR, exist_ok=True)

# 12 features (§3.3) plus 3 HAR features (used only by baseline)
FEATURE_COLS = [
    'r1','r5','r20','vol20','vol60','ma5_ratio','ma20_ratio',
    'rsi','vol_pct','vix','vix_pct','tnx_pct',
]
HAR_COLS = ['rv_d','rv_w','rv_m']


# ---------- Data ----------
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
    Ub = U.ewm(alpha=1/n, adjust=False).mean()
    Db = D.ewm(alpha=1/n, adjust=False).mean()
    f['rsi'] = 100.0 * (1.0 - 1.0 / (1.0 + Ub / (Db + 1e-9)))

    f['vol_pct'] = volume / (volume.rolling(20).mean() + 1e-9) - 1.0

    macro_ff = macro.reindex(f.index, method='ffill')
    f['vix']     = macro_ff['vix']
    f['vix_pct'] = macro_ff['vix'].pct_change(1)
    f['tnx_pct'] = macro_ff['tnx'].pct_change(1)

    # HAR features (Corsi 2009), daily squared-return proxies
    r2 = f['r1'] ** 2
    f['rv_d'] = r2
    f['rv_w'] = r2.rolling(5).mean()
    f['rv_m'] = r2.rolling(22).mean()

    f['log_ret'] = f['r1']   # for target construction

    f = f.replace([np.inf, -np.inf], np.nan).dropna()
    return f


def build_windows_for_asset(feat, T, H, asset_id):
    """
    Per-asset windows (§2.5). Returns list of dicts.
    Targets:
      v_total = sqrt( mean_{h=1..H} r_{t+h}^2 )
      v_down  = sqrt( mean_{h=1..H} min(r_{t+h},0)^2 )
    """
    X_all   = feat[FEATURE_COLS].values
    har_all = feat[HAR_COLS].values
    log_ret = feat['log_ret'].values
    dates   = feat.index
    n       = len(feat)
    out = []
    for i in range(T - 1, n - H):
        Xw    = X_all[i - T + 1 : i + 1].astype(np.float32)
        harw  = har_all[i].astype(np.float32)
        r     = log_ret[i + 1 : i + H + 1]
        v_tot = float(np.sqrt(np.mean(r * r)))
        r_dn  = np.minimum(r, 0.0)
        v_dn  = float(np.sqrt(np.mean(r_dn * r_dn)))
        out.append({
            'X': Xw, 'har': harw,
            'v_total': v_tot, 'v_down': v_dn,
            'date': dates[i], 'asset': asset_id,
        })
    return out


# ---------- Model ----------
class DualHeadTLSTMVol(nn.Module):
    """§7: Head A returns (μ, log σ) for log-vol. §8: Head B returns ℓ̂_down."""
    def __init__(self, n_features, hidden=64, num_layers=1,
                 dropout=0.1, c_sigma=6.0):
        super().__init__()
        self.c_sigma = c_sigma
        self.lstm = nn.LSTM(n_features, hidden, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head_a = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, 2))    # (μ, log σ)
        self.head_b = nn.Sequential(
            nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x, detach_b=False):
        out, _ = self.lstm(x)
        h = self.drop(out[:, -1, :])
        a = self.head_a(h)
        mu    = a[:, 0]
        log_s = torch.clamp(a[:, 1], -self.c_sigma, self.c_sigma)
        h_b   = h.detach() if detach_b else h
        l_dn  = self.head_b(h_b).squeeze(-1)
        return mu, log_s, l_dn


# ---------- Losses ----------
def crps_gaussian_scalar(y, mu, log_sigma):
    """Gaussian CRPS on a scalar target (§9.1)."""
    sigma = torch.exp(log_sigma).clamp(min=1e-6)
    z   = (y - mu) / sigma
    phi = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def composite_loss(mu, log_s, l_dn, l_total, l_down, lam_B, lam_C):
    # §9.5 Head A: CRPS on log total vol
    L_A = crps_gaussian_scalar(l_total, mu, log_s).mean()
    # §9.8 Head B: MSE on log downside semivol
    L_B = F.mse_loss(l_dn, l_down)
    # §9.9 Consistency: μ_A ≈ sg[ℓ_down + ½ log 2]
    if lam_C > 0:
        L_C = F.mse_loss(mu, (l_dn.detach() + 0.5 * math.log(2.0)))
    else:
        L_C = torch.tensor(0.0, device=mu.device)
    total = L_A + lam_B * L_B + lam_C * L_C
    return total, float(L_A), float(L_B), float(L_C)


# ---------- Training ----------
def train_model(model, train_loader, val_loader, lam_B, lam_C,
                epochs, lr, gamma, patience, detach_b=False):
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=gamma)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val, best_state, wait = float('inf'), None, 0
    history = {'train': [], 'val': []}

    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for xb, ltb, ldb in train_loader:
            xb, ltb, ldb = xb.to(DEVICE), ltb.to(DEVICE), ldb.to(DEVICE)
            mu, ls, lv = model(xb, detach_b=detach_b)
            loss, _, _, _ = composite_loss(mu, ls, lv, ltb, ldb, lam_B, lam_C)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb); n += len(xb)
        train_loss = tot / n
        history['train'].append(train_loss)

        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for xb, ltb, ldb in val_loader:
                xb, ltb, ldb = xb.to(DEVICE), ltb.to(DEVICE), ldb.to(DEVICE)
                mu, ls, lv = model(xb, detach_b=detach_b)
                loss, _, _, _ = composite_loss(mu, ls, lv, ltb, ldb, lam_B, lam_C)
                tot += loss.item() * len(xb); n += len(xb)
        val_loss = tot / n
        history['val'].append(val_loss)
        sched.step()

        improved = val_loss < best_val - 1e-5
        if improved:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if ep % 10 == 0 or improved:
            print(f"  ep {ep+1:3d} | train={train_loss:.4f}  val={val_loss:.4f}"
                  f"{' *' if improved else ''}")
        if wait >= patience:
            print(f"  early stop ep {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_val


@torch.no_grad()
def predict(model, X, detach_b=False, bs=2048):
    model.eval()
    Xt = torch.as_tensor(X, dtype=torch.float32)
    mus, sigs, lvs = [], [], []
    for i in range(0, len(Xt), bs):
        xb = Xt[i:i+bs].to(DEVICE)
        mu, ls, lv = model(xb, detach_b=detach_b)
        mus.append(mu.cpu().numpy())
        sigs.append(torch.exp(ls).cpu().numpy())
        lvs.append(lv.cpu().numpy())
    return np.concatenate(mus), np.concatenate(sigs), np.concatenate(lvs)


# ---------- Metrics ----------
def qlike(l_true, mu_pred):
    """QLIKE: exp(u) - u - 1 with u = ℓ_true - μ. Minimized at 0, penalizes underestimation."""
    u = l_true - mu_pred
    return float((np.exp(u) - u - 1.0).mean())


def gauss_interval_coverage(l_true, mu, sigma, level):
    """Empirical coverage of the Gaussian [level]-interval on log-vol."""
    from scipy.stats import norm
    z = norm.ppf(0.5 + level / 2.0)
    lo, hi = mu - z * sigma, mu + z * sigma
    return float(((l_true >= lo) & (l_true <= hi)).mean())


def full_metrics(l_true, mu_pred, sigma_pred, v_true, label):
    crps = float(crps_gaussian_scalar(
        torch.tensor(l_true), torch.tensor(mu_pred), torch.log(torch.tensor(sigma_pred))
    ).mean())
    pearson  = float(np.corrcoef(np.exp(mu_pred), v_true)[0, 1])
    spearman = float(pd.Series(mu_pred).corr(pd.Series(l_true), method='spearman'))
    ql       = qlike(l_true, mu_pred)
    mse_log  = float(np.mean((mu_pred - l_true) ** 2))
    cov80    = gauss_interval_coverage(l_true, mu_pred, sigma_pred, 0.80)
    cov95    = gauss_interval_coverage(l_true, mu_pred, sigma_pred, 0.95)
    print(f"  {label:<30} CRPS={crps:.4f}  ρ={pearson:.4f}  ρ_s={spearman:.4f}  "
          f"QLIKE={ql:.4f}  MSE_ℓ={mse_log:.4f}  cov80={cov80:.3f}  cov95={cov95:.3f}")
    return dict(crps=crps, pearson=pearson, spearman=spearman,
                qlike=ql, mse_log=mse_log, cov80=cov80, cov95=cov95)


# ==================================================================
# Main
# ==================================================================
def main():
    print(f"\nConfig: T={T}, H={H}, hidden={HIDDEN}, "
          f"λ_B={LAMBDA_B}, λ_C={LAMBDA_C}")

    # ---------- Data ----------
    print("\n[1/7] Downloading data...")
    macro = download_macro(START_DATE, END_DATE)
    assets = {}
    for tk in TRAIN_TICKERS:
        raw = download_prices(tk, START_DATE, END_DATE)
        if raw is None: continue
        feat = compute_features(raw, macro)
        if len(feat) < T + H + 200: continue
        assets[tk] = feat
        print(f"  {tk}: {len(feat)} rows")
    print(f"  Loaded {len(assets)} assets")

    # ---------- Windows ----------
    print("\n[2/7] Building per-asset windows...")
    all_w = []
    for tk, feat in assets.items():
        all_w.extend(build_windows_for_asset(feat, T, H, tk))
    print(f"  Total windows: {len(all_w)}")

    # ---------- Calendar split with embargo ----------
    print("\n[3/7] Calendar-date split (embargo T+H)...")
    master = sorted(assets['SPY'].index)
    m = len(master)
    t1 = int(0.60 * m); t2 = int(0.80 * m)
    emb = T + H
    train_end_date  = master[t1 - emb - 1]
    val_start_date  = master[t1]
    val_end_date    = master[t2 - emb - 1]
    test_start_date = master[t2]

    train_w = [w for w in all_w if w['date'] <= train_end_date]
    val_w   = [w for w in all_w if val_start_date <= w['date'] <= val_end_date]
    test_w  = [w for w in all_w if w['date'] >= test_start_date]
    print(f"  Train ends {train_end_date.date()}, val in "
          f"[{val_start_date.date()}, {val_end_date.date()}], "
          f"test from {test_start_date.date()}")
    print(f"  Windows: train={len(train_w)}, val={len(val_w)}, test={len(test_w)}")

    def stack(ws, key): return np.array([w[key] for w in ws], dtype=np.float32)

    train_X   = np.stack([w['X'] for w in train_w]).astype(np.float32)
    val_X     = np.stack([w['X'] for w in val_w]).astype(np.float32)
    test_X    = np.stack([w['X'] for w in test_w]).astype(np.float32)

    train_har = np.stack([w['har'] for w in train_w]).astype(np.float32)
    val_har   = np.stack([w['har'] for w in val_w]).astype(np.float32)
    test_har  = np.stack([w['har'] for w in test_w]).astype(np.float32)

    train_vt  = np.array([w['v_total'] for w in train_w], dtype=np.float32)
    val_vt    = np.array([w['v_total'] for w in val_w], dtype=np.float32)
    test_vt   = np.array([w['v_total'] for w in test_w], dtype=np.float32)

    train_vd  = np.array([w['v_down'] for w in train_w], dtype=np.float32)
    val_vd    = np.array([w['v_down'] for w in val_w], dtype=np.float32)
    test_vd   = np.array([w['v_down'] for w in test_w], dtype=np.float32)

    # ---------- Log targets ----------
    EPS = 1e-6
    train_lt = np.log(train_vt + EPS); val_lt = np.log(val_vt + EPS); test_lt = np.log(test_vt + EPS)
    train_ld = np.log(train_vd + EPS); val_ld = np.log(val_vd + EPS); test_ld = np.log(test_vd + EPS)

    # ---------- Standardize features on train ----------
    print("\n[4/7] Fitting train-fold scaler...")
    mu_f = train_X.mean(axis=(0, 1)).astype(np.float32)
    sd_f = (train_X.std(axis=(0, 1)) + 1e-8).astype(np.float32)
    train_X = (train_X - mu_f) / sd_f
    val_X   = (val_X   - mu_f) / sd_f
    test_X  = (test_X  - mu_f) / sd_f
    print(f"  Scaler fit on {len(train_X)} windows")

    # ---------- Loaders ----------
    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_X), torch.tensor(train_lt),
                      torch.tensor(train_ld)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(val_X), torch.tensor(val_lt),
                      torch.tensor(val_ld)),
        batch_size=BATCH_SIZE, shuffle=False)

    # ---------- HAR baseline ----------
    print("\n[5/7] Fitting HAR-RV baseline (Corsi 2009)...")
    Xh_tr = np.log(train_har + EPS)
    Xh_va = np.log(val_har + EPS)
    Xh_te = np.log(test_har + EPS)
    har = LinearRegression().fit(Xh_tr, train_lt)
    har_val_pred  = har.predict(Xh_va)
    har_test_pred = har.predict(Xh_te)
    # HAR variance = residual variance on train
    har_val_sigma = np.full(len(har_val_pred), np.std(train_lt - har.predict(Xh_tr)))
    har_test_sigma = np.full(len(har_test_pred), har_val_sigma[0])
    print(f"  HAR coefs: {har.coef_}, intercept={har.intercept_:.4f}")
    print(f"  HAR train R²: {har.score(Xh_tr, train_lt):.4f}")

    # ---------- Train main model ----------
    print("\n[6/7] Training main Dual-Head TLSTM (vol)...")
    model = DualHeadTLSTMVol(len(FEATURE_COLS), HIDDEN, NUM_LAYERS,
                             DROPOUT, C_SIGMA).to(DEVICE)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    t0 = time.time()
    hist_main, best_val = train_model(
        model, train_loader, val_loader,
        LAMBDA_B, LAMBDA_C, EPOCHS, LR, GAMMA, PATIENCE, detach_b=False)
    print(f"  Time: {(time.time()-t0)/60:.1f} min | best val = {best_val:.4f}")

    # ---------- Test predictions ----------
    print("\n[7/7] Evaluating on test...")
    test_mu, test_sig, test_ldn = predict(model, test_X)

    # ---------- Baselines ----------
    # B1: constant (train mean log vol)
    const_mu = np.full(len(test_lt), train_lt.mean())
    const_sig = np.full(len(test_lt), train_lt.std())

    # B3: single-head
    print("\n  Training single-head baseline (λ_B=λ_C=0)...")
    torch.manual_seed(SEED); np.random.seed(SEED)
    m_sh = DualHeadTLSTMVol(len(FEATURE_COLS), HIDDEN, NUM_LAYERS,
                            DROPOUT, C_SIGMA).to(DEVICE)
    train_model(m_sh, train_loader, val_loader,
                0.0, 0.0, EPOCHS, LR, GAMMA, PATIENCE, detach_b=False)
    sh_mu, sh_sig, _ = predict(m_sh, test_X)

    # B4: detached Head B
    print("  Training detached Head B baseline...")
    torch.manual_seed(SEED); np.random.seed(SEED)
    m_dt = DualHeadTLSTMVol(len(FEATURE_COLS), HIDDEN, NUM_LAYERS,
                            DROPOUT, C_SIGMA).to(DEVICE)
    train_model(m_dt, train_loader, val_loader,
                LAMBDA_B, LAMBDA_C, EPOCHS, LR, GAMMA, PATIENCE, detach_b=True)
    dt_mu, dt_sig, _ = predict(m_dt, test_X, detach_b=True)

    # ---------- Metrics ----------
    print("\n" + "=" * 110)
    print(f"{'Model':<30} {'CRPS':>8} {'Pearson':>9} {'Spear':>8} "
          f"{'QLIKE':>8} {'MSE_ℓ':>8} {'cov80':>7} {'cov95':>7}")
    print("-" * 110)
    res = {}
    res['Main dual-head'] = full_metrics(test_lt, test_mu, test_sig, test_vt, 'Main dual-head')
    res['HAR-RV']         = full_metrics(test_lt, har_test_pred, har_test_sigma, test_vt, 'HAR-RV (Corsi)')
    res['Single-head']    = full_metrics(test_lt, sh_mu, sh_sig, test_vt, 'Single-head')
    res['Detached Head B']= full_metrics(test_lt, dt_mu, dt_sig, test_vt, 'Detached Head B')
    res['Constant']       = full_metrics(test_lt, const_mu, const_sig, test_vt, 'Constant')
    print("=" * 110)

    # ---------- Abstention diagnostic ----------
    val_mu, val_sig, _ = predict(model, val_X)
    tau_sigma = float(np.quantile(val_sig, KAPPA_0))
    act = test_sig <= tau_sigma
    print(f"\nAbstention diagnostic (τ_σ = q80 val = {tau_sigma:.4f}):")
    print(f"  Coverage: {act.mean():.3f}")
    if act.sum() > 10:
        conf_metrics = full_metrics(test_lt[act], test_mu[act], test_sig[act],
                                    test_vt[act], 'Confident subset')
        unc_metrics  = full_metrics(test_lt[~act], test_mu[~act], test_sig[~act],
                                    test_vt[~act], 'Uncertain subset')
        print(f"  → σ̂ correctly identifies easy vs hard windows"
              if conf_metrics['crps'] < unc_metrics['crps'] else
              f"  → σ̂ does NOT separate easy from hard")

    # ---------- Save ----------
    torch.save({
        'state_dict': model.state_dict(),
        'config': {
            'n_features': len(FEATURE_COLS), 'hidden': HIDDEN,
            'num_layers': NUM_LAYERS, 'dropout': DROPOUT,
            'window': T, 'horizon': H, 'c_sigma': C_SIGMA,
        },
        'feature_cols': FEATURE_COLS,
        'mu_f': mu_f, 'sd_f': sd_f, 'eps': EPS,
        'har_coef': har.coef_, 'har_intercept': har.intercept_,
        'har_sigma': float(har_val_sigma[0]),
        'history': hist_main,
        'test_metrics': res,
    }, CKPT_PATH)
    print(f"\nSaved: {CKPT_PATH}")

    # ---------- Plots ----------
    fig = plt.figure(figsize=(20, 12))
    gs  = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)

    # 1. Loss curves
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(hist_main['train'], label='train')
    ax.plot(hist_main['val'], label='val')
    ax.set(xlabel='epoch', ylabel='loss', title='Loss (main model)')
    ax.legend(); ax.grid(alpha=0.3)

    # 2. Predicted vs realized (log space)
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(test_mu, test_lt, s=2, alpha=0.2)
    lim = [min(test_mu.min(), test_lt.min()),
           max(test_mu.max(), test_lt.max())]
    ax.plot(lim, lim, 'r--', lw=1)
    ax.set(xlabel='μ̂ (pred log-vol)', ylabel='ℓ^total (realized log-vol)',
           title=f'Log-vol scatter (ρ={res["Main dual-head"]["pearson"]:.3f})')
    ax.grid(alpha=0.3)

    # 3. Reliability: interval coverage
    ax = fig.add_subplot(gs[0, 2])
    levels = np.array([0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95])
    covs = [gauss_interval_coverage(test_lt, test_mu, test_sig, lv) for lv in levels]
    ax.plot(levels, covs, 'o-', label='Empirical')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Ideal')
    ax.set(xlabel='Nominal coverage', ylabel='Empirical coverage',
           title='Interval calibration')
    ax.legend(); ax.grid(alpha=0.3)

    # 4. CRPS by predicted-vol decile (does σ̂ identify hard cases?)
    ax = fig.add_subplot(gs[1, 0])
    order = np.argsort(test_sig)
    deciles = np.array_split(order, 10)
    crps_per_decile = []
    for d in deciles:
        c = crps_gaussian_scalar(
            torch.tensor(test_lt[d]),
            torch.tensor(test_mu[d]),
            torch.log(torch.tensor(test_sig[d]))
        ).mean().item()
        crps_per_decile.append(c)
    ax.bar(range(1, 11), crps_per_decile, color='tab:purple')
    ax.set(xlabel='predicted σ̂ decile (1=confident)', ylabel='CRPS',
           title='CRPS vs σ̂ decile')
    ax.grid(alpha=0.3)

    # 5. Model comparison table
    ax = fig.add_subplot(gs[1, 1]); ax.axis('off')
    rows = []
    for name in ['Main dual-head', 'HAR-RV', 'Single-head',
                 'Detached Head B', 'Constant']:
        r = res[name]
        rows.append([name, f"{r['crps']:.4f}", f"{r['pearson']:.4f}",
                     f"{r['qlike']:.4f}", f"{r['mse_log']:.4f}"])
    cols = ['Model', 'CRPS', 'ρ', 'QLIKE', 'MSE_ℓ']
    t = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1.1, 1.5)
    ax.set_title('Test metrics', pad=10)

    # 6. Per-asset Pearson
    ax = fig.add_subplot(gs[1, 2])
    asset_ids = [w['asset'] for w in test_w]
    pers = []
    for a in sorted(set(asset_ids)):
        m = np.array(asset_ids) == a
        p = float(np.corrcoef(np.exp(test_mu[m]), test_vt[m])[0, 1])
        pers.append((a, p))
    pers.sort(key=lambda x: x[1])
    names = [p[0] for p in pers]; vals = [p[1] for p in pers]
    colors = ['tab:red' if v < 0.5 else 'tab:green' for v in vals]
    ax.barh(names, vals, color=colors)
    ax.axvline(0.5, ls='--', color='gray')
    ax.set(xlabel='Pearson ρ(exp(μ̂), v^total)',
           title='Per-asset Pearson correlation')
    ax.grid(alpha=0.3)

    plt.suptitle(f'Dual-Head TLSTM Vol v3.0 — {len(assets)} assets, '
                 f'T={T}, H={H}, λ_B={LAMBDA_B}, λ_C={LAMBDA_C}',
                 fontsize=13, y=0.995)
    plt.savefig('dual_head_tlstm_vol_v3.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("Saved: dual_head_tlstm_vol_v3.png")


if __name__ == "__main__":
    main()