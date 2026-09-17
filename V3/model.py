"""
Dual-Head TLSTM v3.0 — multi-asset training, multi-asset testing
================================================================
Train on 40 liquid US ETFs + mega-caps.
Zero-shot test on 23 held-out assets across classes.

Strict v2.0 causality:
  - Embargoed chronological split (Def 2.9)
  - Per-asset windows (Spec 2.13)
  - Pooled train-fold scaler (Lemma 4.2)
  - Forward-fill only for macro (Prop 2.7)
  - Stop-gradient on Head B target (Rem 9.2)
  - Isotonic on validation only (Thm 12.2)
"""

import os, warnings, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ------------------------------------------------------------------
# Universe
# ------------------------------------------------------------------
TRAIN_TICKERS = [
    'SPY','QQQ','IWM','DIA','MDY','RSP','VTI',
    'XLE','XLF','XLK','XLV','XLI','XLP','XLY','XLU','XLB','XLRE','XLC',
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

# ------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------
T_WINDOW   = 30
HIDDEN     = 96
NUM_LAYERS = 2
DROPOUT    = 0.3
BATCH_SIZE = 512
EPOCHS     = 80
LR         = 3e-4
GAMMA      = 3e-4          # AdamW weight decay
LAMBDA     = 0.3           # auxiliary loss weight
PATIENCE   = 15
TAU_DEPLOY = 0.55          # lowered; matches realistic model output scale
N_SEEDS    = 3
CACHE_DIR  = './cache_v3'
os.makedirs(CACHE_DIR, exist_ok=True)

FEATURE_COLS = [
    'r1','r3','r20','vol20','vol60','ma5_ratio','ma20_ratio',
    'rsi','vol_pct','vix','vix_pct','tnx','tnx_pct','term_spread'
]

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
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
    if df is None or len(df) == 0:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if 'Close' not in df.columns or 'Volume' not in df.columns:
        return None
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

    f['r1']  = np.log(close).diff(1)
    f['r3']  = np.log(close).diff(3)
    f['r20'] = np.log(close).diff(20)
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
    f['vix'] = macro_ff['vix']
    f['vix_pct'] = macro_ff['vix'].pct_change(1)
    f['tnx'] = macro_ff['tnx']
    f['tnx_pct'] = macro_ff['tnx'].pct_change(1)
    f['term_spread'] = macro_ff['tnx'] - macro_ff['vix'] * 0.1

    f['target']      = (close.shift(-1) > close).astype(float)
    f['next_return'] = close.shift(-1) / close - 1.0

    f = f.replace([np.inf, -np.inf], np.nan).dropna()
    return f


def build_windows(features, targets, window):
    X, y = [], []
    for i in range(window - 1, len(features)):
        X.append(features[i - window + 1 : i + 1])
        y.append(targets[i])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
class DualHeadTLSTM(nn.Module):
    def __init__(self, n_features, hidden=96, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head_a = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_b = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.drop(out.mean(dim=1))
        return self.head_a(h).squeeze(-1), self.head_b(h).squeeze(-1)


# ------------------------------------------------------------------
# Loss / helpers
# ------------------------------------------------------------------
def composite_loss(z_a, z_b, y, lam):
    L_A = F.binary_cross_entropy_with_logits(z_a, y)
    with torch.no_grad():
        pred_a = (torch.sigmoid(z_a) >= 0.5).float()
        correct = (pred_a == y).float()
    L_B = F.binary_cross_entropy_with_logits(z_b, correct)
    return L_A + lam * L_B


def train_epoch(model, loader, opt, lam):
    model.train(); tot, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        z_a, z_b = model(xb)
        loss = composite_loss(z_a, z_b, yb, lam)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item() * len(xb); n += len(xb)
    return tot / n


@torch.no_grad()
def evaluate(model, loader, lam):
    model.eval(); ps, ys, tot, n = [], [], 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        z_a, z_b = model(xb)
        loss = composite_loss(z_a, z_b, yb, lam)
        p_a = torch.sigmoid(z_a); q = torch.sigmoid(z_b)
        ps.append((q*p_a + (1-q)*(1-p_a)).cpu().numpy())
        ys.append(yb.cpu().numpy())
        tot += loss.item() * len(xb); n += len(xb)
    return np.concatenate(ps), np.concatenate(ys), tot / n


@torch.no_grad()
def predict_final(model, X, bs=2048):
    model.eval(); ps = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i+bs]).to(DEVICE)
        z_a, z_b = model(xb)
        p_a = torch.sigmoid(z_a); q = torch.sigmoid(z_b)
        ps.append((q*p_a + (1-q)*(1-p_a)).cpu().numpy())
    return np.concatenate(ps)


@torch.no_grad()
def predict_head_a(model, X, bs=2048):
    model.eval(); ps = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i+bs]).to(DEVICE)
        z_a, _ = model(xb)
        ps.append(torch.sigmoid(z_a).cpu().numpy())
    return np.concatenate(ps)


def ece(p, y, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0: continue
        e += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return e


def selective_metrics(p_cal, y, tau):
    act = (p_cal >= tau) | (p_cal <= 1 - tau)
    kappa = act.mean()
    if act.sum() < 5:
        return kappa, np.nan
    pred = np.zeros_like(y)
    pred[p_cal >= tau] = 1
    return kappa, (pred[act] == y[act]).mean()


# ==================================================================
# Pipeline
# ==================================================================

# ---------- [1] Download macro -------------------------------------
print("\n[1/7] Downloading macro series...")
macro = download_macro(START_DATE, END_DATE)
print(f"  Macro: {macro.shape}, columns={list(macro.columns)}")

# ---------- [2] Download and featurize all training assets ---------
print(f"\n[2/7] Downloading {len(TRAIN_TICKERS)} training assets...")
train_data = {}
for tk in TRAIN_TICKERS:
    raw = download_prices(tk, START_DATE, END_DATE)
    if raw is None or len(raw) < 500:
        print(f"  {tk}: skip (insufficient)"); continue
    feat = compute_features(raw, macro)
    if len(feat) < 500:
        print(f"  {tk}: skip (insufficient after featurization)"); continue
    train_data[tk] = feat
    print(f"  {tk}: {len(feat)} rows")

print(f"\n  Loaded {len(train_data)} training assets")

# ---------- [3] Pooled train/val/test split across assets ----------
# Use the same calendar bounds for every asset, so all assets have the
# same folds. This is the standard cross-sectional setup.

common_start = max(f.index.min() for f in train_data.values())
common_end   = min(f.index.max() for f in train_data.values())
print(f"\n  Common date range: {common_start.date()} to {common_end.date()}")

# For each asset, get aligned feature matrix over the common range
aligned = {}
for tk, f in train_data.items():
    fa = f.loc[common_start:common_end]
    if len(fa) < 300: continue
    aligned[tk] = fa

n_days = min(len(f) for f in aligned.values())
# Relabel to a shared integer time index 0..n_days-1
for tk in aligned:
    aligned[tk] = aligned[tk].iloc[:n_days].copy()
    aligned[tk]['t_idx'] = np.arange(n_days)

T1 = int(0.60 * n_days)
T2 = int(0.80 * n_days)
emb = T_WINDOW

# Train-fold pooled statistics (Lemma 4.2, Spec 4.1)
train_block = np.concatenate(
    [aligned[tk][FEATURE_COLS].values[:T1 - emb] for tk in aligned],
    axis=0
)
mu = train_block.mean(axis=0)
sd = train_block.std(axis=0)
print(f"  Pooled scaler fit on {train_block.shape[0]} rows across {len(aligned)} assets")

# Build per-asset windows (Spec 2.13) — never across asset boundaries
def build_split(aligned, FEATURE_COLS, mu, sd, T_WINDOW,
                train_idx, val_idx, test_idx):
    Xtr, ytr = [], []
    Xva, yva = [], []
    Xte, yte = [], []
    for tk, f in aligned.items():
        Xs = (f[FEATURE_COLS].values - mu) / (sd + 1e-8)
        ys = f['target'].values
        t  = f['t_idx'].values
        for i in range(T_WINDOW - 1, len(f)):
            xi = Xs[i - T_WINDOW + 1 : i + 1]
            yi = ys[i]
            ti = t[i]
            if ti <= train_idx[-1]:
                Xtr.append(xi); ytr.append(yi)
            elif val_idx[0] <= ti <= val_idx[-1]:
                Xva.append(xi); yva.append(yi)
            elif test_idx[0] <= ti <= test_idx[-1]:
                Xte.append(xi); yte.append(yi)
    return (np.asarray(Xtr, dtype=np.float32), np.asarray(ytr, dtype=np.float32),
            np.asarray(Xva, dtype=np.float32), np.asarray(yva, dtype=np.float32),
            np.asarray(Xte, dtype=np.float32), np.asarray(yte, dtype=np.float32))

idx_train = np.arange(0, T1 - emb)
idx_val   = np.arange(T1 + 1, T2 - emb)
idx_test  = np.arange(T2 + 1, n_days)

print(f"  Folds: train=[0, {T1-emb}], val=[{T1+1}, {T2-emb}], test=[{T2+1}, {n_days-1}]")

train_X, train_y, val_X, val_y, test_X, test_y = build_split(
    aligned, FEATURE_COLS, mu, sd, T_WINDOW, idx_train, idx_val, idx_test)

print(f"  Windows: train={train_X.shape} val={val_X.shape} test={test_X.shape}")
print(f"  Train pos rate: {train_y.mean():.4f}")
print(f"  Val   pos rate: {val_y.mean():.4f}")
print(f"  Test  pos rate: {test_y.mean():.4f}")

train_loader = DataLoader(
    TensorDataset(torch.tensor(train_X), torch.tensor(train_y)),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader   = DataLoader(
    TensorDataset(torch.tensor(val_X), torch.tensor(val_y)),
    batch_size=BATCH_SIZE, shuffle=False)

# ---------- [4] Train with seed ensemble ---------------------------
print(f"\n[3/7] Training {N_SEEDS}-seed ensemble...")
trained_models = []
val_preds_ens = np.zeros(len(val_y))
val_preds_a_ens = np.zeros(len(val_y))

for seed in range(N_SEEDS):
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    model = DualHeadTLSTM(len(FEATURE_COLS), HIDDEN, NUM_LAYERS, DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=GAMMA)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_val, best_state, wait = np.inf, None, 0
    for ep in range(EPOCHS):
        tr = train_epoch(model, train_loader, opt, LAMBDA)
        p_val, y_val, va = evaluate(model, val_loader, LAMBDA)
        sched.step()
        if va < best_val - 1e-4:
            best_val = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE: break
    model.load_state_dict(best_state)
    trained_models.append(model)

    pv = predict_final(model, val_X)
    pa = predict_head_a(model, val_X)
    val_preds_ens   += pv / N_SEEDS
    val_preds_a_ens += pa / N_SEEDS

    try: auc = roc_auc_score(y_val, pv)
    except Exception: auc = 0.5
    print(f"  seed {seed}: best_val_loss={best_val:.4f}  val_auc={auc:.4f}  "
          f"time={time.time()-t0:.1f}s")

# ---------- [5] Isotonic on validation -----------------------------
iso = IsotonicRegression(out_of_bounds='clip').fit(val_preds_ens, val_y)
val_cal = iso.predict(val_preds_ens)
print(f"\n  Val Brier  raw={np.mean((val_preds_ens-val_y)**2):.4f}"
      f"  cal={np.mean((val_cal-val_y)**2):.4f}")
print(f"  Val ECE    raw={ece(val_preds_ens, val_y):.4f}"
      f"  cal={ece(val_cal, val_y):.4f}")

# ---------- [6] Per-asset zero-shot eval ---------------------------
print(f"\n[4/7] Zero-shot test across {len(TEST_TICKERS)} unseen assets...")
results = {}
for tk in TEST_TICKERS:
    raw = download_prices(tk, START_DATE, END_DATE)
    if raw is None or len(raw) < 500:
        print(f"  {tk}: skip"); continue
    feat = compute_features(raw, macro)
    if len(feat) < 300:
        print(f"  {tk}: skip (short)"); continue

    fa = feat.loc[common_start:common_end]
    if len(fa) < 300:
        print(f"  {tk}: skip (no overlap)"); continue

    Xs = (fa[FEATURE_COLS].values - mu) / (sd + 1e-8)
    ys = fa['target'].values
    rs = fa['next_return'].values

    Xw, yw = build_windows(Xs, ys, T_WINDOW)
    rw = rs[T_WINDOW - 1 : T_WINDOW - 1 + len(yw)]

    # Ensemble
    p_raw = np.zeros(len(yw)); p_a = np.zeros(len(yw))
    for m in trained_models:
        p_raw += predict_final(m, Xw) / len(trained_models)
        p_a   += predict_head_a(m, Xw) / len(trained_models)

    p_cal = iso.predict(p_raw)
    try: auc = roc_auc_score(yw, p_raw)
    except Exception: auc = 0.5
    acc   = ((p_raw >= 0.5).astype(int) == yw).mean()
    brier = np.mean((p_cal - yw)**2)
    kappa, acc_sel = selective_metrics(p_cal, yw, TAU_DEPLOY)

    results[tk] = dict(p_raw=p_raw, p_cal=p_cal, y=yw, ret=rw,
                       acc=acc, auc=auc, brier=brier,
                       kappa=kappa, acc_sel=acc_sel, n=len(yw))
    print(f"  {tk:5s} n={len(yw):5d}  acc={acc:.4f}  auc={auc:.4f}  "
          f"brier={brier:.4f}  cov={kappa:.3f}  "
          f"sel_acc={acc_sel:.4f}" if not np.isnan(acc_sel) else "")

# ---------- [7] Summary --------------------------------------------
print("\n[5/7] Summary")
train_auc = roc_auc_score(train_y, predict_final(trained_models[0], train_X)) \
            if len(np.unique(train_y)) > 1 else 0.5
val_auc   = roc_auc_score(val_y, val_preds_ens) \
            if len(np.unique(val_y)) > 1 else 0.5
test_auc  = roc_auc_score(test_y, np.zeros(len(test_y))) if False else None

# Val predictions on SPY alone for "in-distribution" reference
spy_aligned = aligned['SPY']
Xs_spy = (spy_aligned[FEATURE_COLS].values - mu) / (sd + 1e-8)
ys_spy = spy_aligned['target'].values
Xw_spy, yw_spy = build_windows(Xs_spy, ys_spy, T_WINDOW)
p_spy = np.zeros(len(yw_spy))
for m in trained_models:
    p_spy += predict_final(m, Xw_spy) / len(trained_models)
spy_auc = roc_auc_score(yw_spy, p_spy) if len(np.unique(yw_spy)) > 1 else 0.5

print(f"  Train AUC (pooled): {train_auc:.4f}")
print(f"  Val   AUC (pooled): {val_auc:.4f}")
print(f"  SPY in-sample AUC: {spy_auc:.4f}")

test_aucs = [results[t]['auc'] for t in results]
test_accs = [results[t]['acc'] for t in results]
print(f"  Test  AUC mean±std: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
print(f"  Test  Acc mean±std: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
print(f"  Constant accuracy (majority): {max(test_y.mean(), 1-test_y.mean()):.4f}")

# ==================================================================
# Plots
# ==================================================================
print("\n[6/7] Plotting...")
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.40, wspace=0.30)

# 1. Reliability
ax = fig.add_subplot(gs[0, 0])
bins = np.linspace(0, 1, 11)
def rel(p, y):
    idx = np.digitize(p, bins) - 1
    xs, ys = [], []
    for b in range(10):
        m = idx == b
        if m.sum() > 0:
            xs.append(p[m].mean()); ys.append(y[m].mean())
    return np.array(xs), np.array(ys)
xr, yr = rel(val_preds_ens, val_y)
xc, yc = rel(val_cal, val_y)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
ax.plot(xr, yr, 'o-', color='tab:red',  label='Raw')
ax.plot(xc, yc, 's-', color='tab:blue', label='Calibrated')
ax.set(xlabel='Predicted prob.', ylabel='Empirical freq.',
       title='Reliability (validation, pooled)')
ax.legend(); ax.grid(alpha=0.3)

# 2. Val score distribution
ax = fig.add_subplot(gs[0, 1])
ax.hist(val_preds_ens[val_y == 1], bins=30, alpha=0.5, density=True, label='y=1')
ax.hist(val_preds_ens[val_y == 0], bins=30, alpha=0.5, density=True, label='y=0')
ax.set(xlabel='p_final (raw)', ylabel='Density',
       title='Val score distribution (pooled)')
ax.legend(); ax.grid(alpha=0.3)

# 3. Per-asset AUC bar chart
ax = fig.add_subplot(gs[0, 2])
tk_list = list(results.keys())
aucs = [results[t]['auc'] for t in tk_list]
colors = ['tab:green' if a > 0.5 else 'tab:red' for a in aucs]
ax.barh(tk_list, aucs, color=colors, alpha=0.75)
ax.axvline(0.5, ls='--', color='gray')
ax.set(xlabel='AUC', title='Zero-shot AUC per test asset')
ax.grid(alpha=0.3)

# 4. Test accuracy vs AUC
ax = fig.add_subplot(gs[1, 0])
accs = [results[t]['acc'] for t in tk_list]
ax.scatter(aucs, accs, s=80, alpha=0.7)
for i, tk in enumerate(tk_list):
    ax.annotate(tk, (aucs[i], accs[i]), fontsize=7,
                xytext=(3, 3), textcoords='offset points')
ax.axvline(0.5, ls='--', color='gray', alpha=0.5)
ax.axhline(0.5, ls='--', color='gray', alpha=0.5)
ax.set(xlabel='AUC', ylabel='Accuracy', title='Test: AUC vs accuracy')
ax.grid(alpha=0.3)

# 5. Coverage vs selective accuracy
ax = fig.add_subplot(gs[1, 1])
taus = np.linspace(0.50, 0.90, 17)
for tk in tk_list:
    r = results[tk]
    cov, sa = [], []
    for tau in taus:
        k, a = selective_metrics(r['p_cal'], r['y'], tau)
        cov.append(k); sa.append(a)
    ax.plot(cov, sa, 'o-', label=tk, alpha=0.6, markersize=3)
ax.axhline(0.5, ls='--', color='gray', alpha=0.5)
ax.set(xlabel='Coverage κ', ylabel='Selective accuracy',
       title='Coverage vs selective accuracy')
ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)

# 6. Brier per asset
ax = fig.add_subplot(gs[1, 2])
briers = [results[t]['brier'] for t in tk_list]
ax.barh(tk_list, briers, color='tab:purple', alpha=0.75)
ax.axvline(0.25, ls='--', color='gray', label='Brier=0.25 (chance)')
ax.set(xlabel='Brier', title='Calibrated Brier per test asset')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 7. Training progress (last seed)
ax = fig.add_subplot(gs[2, 0])
ax.axis('off')
lines = [
    f"Train AUC (pooled):    {train_auc:.4f}",
    f"Val   AUC (pooled):    {val_auc:.4f}",
    f"SPY in-sample AUC:     {spy_auc:.4f}",
    f"Test  AUC  mean±std:   {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}",
    f"Test  Acc  mean±std:   {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}",
    f"Test  Brier mean:      {np.mean([results[t]['brier'] for t in tk_list]):.4f}",
    f"",
    f"#assets trained: {len(aligned)}",
    f"#windows train: {len(train_y)}",
    f"#windows val:   {len(val_y)}",
    f"#windows test/asset: {len(yw):,}",
    f"#params:        {sum(p.numel() for p in trained_models[0].parameters()):,}",
]
for i, line in enumerate(lines):
    ax.text(0, 1 - i * 0.075, line, fontsize=10, family='monospace',
            transform=ax.transAxes)
ax.set_title('Summary', pad=10)

# 8. Per-class prediction histograms
ax = fig.add_subplot(gs[2, 1])
all_p, all_y = [], []
for tk in tk_list:
    r = results[tk]
    all_p.append(r['p_cal']); all_y.append(r['y'])
all_p = np.concatenate(all_p); all_y = np.concatenate(all_y)
ax.hist(all_p[all_y == 1], bins=40, alpha=0.5, density=True, label='y=1')
ax.hist(all_p[all_y == 0], bins=40, alpha=0.5, density=True, label='y=0')
ax.axvline(0.5, ls='--', color='k')
ax.set(xlabel='p_cal', ylabel='Density',
       title='Pooled test: score distribution by class')
ax.legend(); ax.grid(alpha=0.3)

# 9. Selective accuracy at τ
ax = fig.add_subplot(gs[2, 2])
sa_tau = [results[t]['acc_sel'] for t in tk_list]
cov_tau = [results[t]['kappa'] for t in tk_list]
x = np.arange(len(tk_list)); w = 0.35
ax.bar(x - w/2, cov_tau, w, label='Coverage', color='tab:purple')
ax.bar(x + w/2, sa_tau, w, label='Sel. acc',  color='tab:green')
ax.axhline(TAU_DEPLOY, ls='--', color='k', alpha=0.6, label=f'τ={TAU_DEPLOY}')
ax.set_xticks(x); ax.set_xticklabels(tk_list, rotation=45, fontsize=7)
ax.set(title=f'Coverage / selective accuracy at τ={TAU_DEPLOY}')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.suptitle(f'Dual-Head TLSTM v3.0 — trained on {len(aligned)} assets, '
             f'zero-shot on {len(tk_list)} unseen assets',
             fontsize=13, y=0.995)
plt.savefig('dual_head_tlstm_v3.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved dual_head_tlstm_v3.png")

# ---------- CSV summary --------------------------------------------
summary = pd.DataFrame([
    {'ticker': tk, 'n': results[tk]['n'],
     'accuracy': results[tk]['acc'], 'auc': results[tk]['auc'],
     'brier': results[tk]['brier'], 'coverage': results[tk]['kappa'],
     'selective_accuracy': results[tk]['acc_sel']}
    for tk in tk_list
])
summary.loc[len(summary)] = {
    'ticker': 'MEAN', 'n': int(np.mean([results[t]['n'] for t in tk_list])),
    'accuracy': np.mean(test_accs), 'auc': np.mean(test_aucs),
    'brier': np.mean([results[t]['brier'] for t in tk_list]),
    'coverage': np.mean(cov_tau),
    'selective_accuracy': np.nanmean(sa_tau)}
summary.to_csv('dual_head_tlstm_v3_summary.csv', index=False)
print("Saved dual_head_tlstm_v3_summary.csv")

print("\n[7/7] Done.")