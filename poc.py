"""
Dual Head TLSTM - Proof of Concept
==================================
Train on SPY, zero-shot on QQQ, IWM, DIA, GLD, TLT.
Next-day direction with error-correction and selective abstention.

Strict no-leakage protocol:
  - chronological train/val split within the training asset
  - scaler fit on training fold only
  - windows built per-asset (no cross-boundary windows)
  - isotonic calibration fit on validation only
  - single evaluation on unseen assets
"""

import os, warnings
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
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
TRAIN_TICKER = 'SPY'
TEST_TICKERS = ['QQQ', 'IWM', 'DIA', 'GLD', 'TLT']
START_DATE   = '2010-01-01'
END_DATE     = '2023-12-31'
WINDOW       = 30
HIDDEN       = 32
NUM_LAYERS   = 1
BATCH_SIZE   = 64
EPOCHS       = 40
LR           = 1e-3
LAMBDA       = 1.0
PATIENCE     = 8
TAU_DEPLOY   = 0.60
CACHE_DIR    = './cache'
os.makedirs(CACHE_DIR, exist_ok=True)

FEATURE_COLS = ['r1','r3','vol20','ma5_ratio','ma20_ratio','rsi','vol_chg']

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
def download_data(ticker, start, end):
    cache_path = os.path.join(CACHE_DIR, f"{ticker}_{start}_{end}.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Close','Volume']].dropna()
    try: df.to_parquet(cache_path)
    except Exception: pass
    return df


def compute_features(df):
    """All features use data <= t. Labels use t+1 only as supervision."""
    close  = df['Close'].astype(float)
    volume = df['Volume'].astype(float)

    f = pd.DataFrame(index=df.index)
    f['r1']          = close.pct_change(1)
    f['r3']          = close.pct_change(3)
    f['vol20']       = f['r1'].rolling(20).std()
    f['ma5_ratio']   = close.rolling(5).mean()  / close
    f['ma20_ratio']  = close.rolling(20).mean() / close

    delta = close.diff()
    up    = delta.clip(lower=0).rolling(14).mean()
    down  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = up / (down + 1e-9)
    f['rsi'] = 100 - 100 / (1 + rs)

    f['vol_chg'] = volume.pct_change(1)

    # Labels only (never used as features)
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
    def __init__(self, n_features, hidden=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers, batch_first=True)
        self.head_a = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1))
        self.head_b = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out.mean(dim=1)
        z_a = self.head_a(h).squeeze(-1)
        z_b = self.head_b(h).squeeze(-1)
        return z_a, z_b


# ------------------------------------------------------------------
# Training / eval helpers
# ------------------------------------------------------------------
def train_epoch(model, loader, opt, lam):
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        z_a, z_b = model(xb)
        L_final = F.binary_cross_entropy_with_logits(z_a + z_b, yb)
        L_A     = F.binary_cross_entropy_with_logits(z_a, yb)
        loss    = L_final + lam * L_A
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(xb); n += len(xb)
    return total / n


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ps, ys, tot, n = [], [], 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        z_a, z_b = model(xb)
        p = torch.sigmoid(z_a + z_b)
        loss = F.binary_cross_entropy_with_logits(z_a + z_b, yb)
        ps.append(p.cpu().numpy()); ys.append(yb.cpu().numpy())
        tot += loss.item() * len(xb); n += len(xb)
    return np.concatenate(ps), np.concatenate(ys), tot / n


@torch.no_grad()
def predict(model, X, bs=256):
    model.eval()
    ps = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i+bs]).to(DEVICE)
        z_a, z_b = model(xb)
        ps.append(torch.sigmoid(z_a + z_b).cpu().numpy())
    return np.concatenate(ps)


# ==================================================================
# Pipeline
# ==================================================================

print("\n[1/6] Downloading training asset...")
train_raw  = download_data(TRAIN_TICKER, START_DATE, END_DATE)
train_feat = compute_features(train_raw)
print(f"  {TRAIN_TICKER}: {len(train_feat)} usable rows")

# ---------------- Chronological split, scaler on train only --------
n        = len(train_feat)
n_train  = int(0.70 * n)              # train fold for scaler + model
mu       = train_feat[FEATURE_COLS].iloc[:n_train].mean().values
sigma    = train_feat[FEATURE_COLS].iloc[:n_train].std().values
print(f"  Scaler fit on first {n_train} rows ({n_train/n:.0%}) only")

scaled_full   = (train_feat[FEATURE_COLS].values - mu) / (sigma + 1e-8)
targets_full  = train_feat['target'].values

train_Xr, train_yr = scaled_full[:n_train], targets_full[:n_train]
val_Xr,   val_yr   = scaled_full[n_train:], targets_full[n_train:]

# Per-fold windows (never cross fold boundary)
train_X, train_y = build_windows(train_Xr, train_yr, WINDOW)
val_X,   val_y   = build_windows(val_Xr,   val_yr,   WINDOW)
print(f"  Train windows: {train_X.shape} | Val windows: {val_X.shape}")

train_loader = DataLoader(TensorDataset(torch.tensor(train_X),
                                        torch.tensor(train_y)),
                          batch_size=BATCH_SIZE, shuffle=False)
val_loader   = DataLoader(TensorDataset(torch.tensor(val_X),
                                        torch.tensor(val_y)),
                          batch_size=BATCH_SIZE, shuffle=False)

# ---------------- Model -------------------------------------------
print("\n[2/6] Training Dual Head TLSTM...")
model = DualHeadTLSTM(len(FEATURE_COLS), HIDDEN, NUM_LAYERS).to(DEVICE)
opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}
best_val, best_state, wait = np.inf, None, 0

for ep in range(EPOCHS):
    tr = train_epoch(model, train_loader, opt, LAMBDA)
    p_val, y_val, va = evaluate(model, val_loader)
    acc = ((p_val >= 0.5).astype(int) == y_val).mean()
    try:    auc = roc_auc_score(y_val, p_val)
    except: auc = 0.5

    history['train_loss'].append(tr)
    history['val_loss'].append(va)
    history['val_acc'].append(acc)
    history['val_auc'].append(auc)

    print(f"  Epoch {ep+1:3d} | train={tr:.4f}  val={va:.4f}  "
          f"acc={acc:.4f}  auc={auc:.4f}")

    if va < best_val - 1e-4:
        best_val = va
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"  Early stop at epoch {ep+1}")
            break

if best_state is not None:
    model.load_state_dict(best_state)

# ---------------- Isotonic calibration on validation ---------------
print("\n[3/6] Fitting isotonic calibration on validation only...")
p_val_raw = predict(model, val_X)
iso = IsotonicRegression(out_of_bounds='clip').fit(p_val_raw, val_y)
p_val_cal = iso.predict(p_val_raw)
brier_raw = np.mean((p_val_raw - val_y) ** 2)
brier_cal = np.mean((p_val_cal - val_y) ** 2)
print(f"  Val Brier raw: {brier_raw:.4f} -> calibrated: {brier_cal:.4f}")

# ---------------- Zero-shot evaluation ------------------------------
print("\n[4/6] Zero-shot evaluation on unseen assets...")
results = {}
for tk in TEST_TICKERS:
    try:
        raw  = download_data(tk, START_DATE, END_DATE)
        feat = compute_features(raw)
    except Exception as e:
        print(f"  {tk}: download failed ({e})"); continue
    if len(feat) < WINDOW + 10:
        print(f"  {tk}: not enough rows"); continue

    scaled   = (feat[FEATURE_COLS].values - mu) / (sigma + 1e-8)
    targets  = feat['target'].values
    next_ret = feat['next_return'].values

    X_te, y_te = build_windows(scaled, targets, WINDOW)
    ret_te     = next_ret[WINDOW - 1 : WINDOW - 1 + len(y_te)]

    p_raw = predict(model, X_te)
    p_cal = iso.predict(p_raw)

    acc = ((p_raw >= 0.5).astype(int) == y_te).mean()
    try:    auc = roc_auc_score(y_te, p_raw)
    except: auc = 0.5
    brier = np.mean((p_cal - y_te) ** 2)

    results[tk] = dict(p_raw=p_raw, p_cal=p_cal, y=y_te, ret=ret_te,
                       acc=acc, auc=auc, brier=brier, n=len(y_te))

    print(f"  {tk:4s} n={len(y_te):5d}  acc={acc:.4f}  auc={auc:.4f}  "
          f"brier={brier:.4f}")

# ---------------- Selective curves ----------------------------------
print("\n[5/6] Selective prediction curves...")
taus   = np.linspace(0.50, 0.95, 19)
curves = {}
for tk, r in results.items():
    c = []
    for tau in taus:
        act = (r['p_cal'] >= tau) | (r['p_cal'] <= 1 - tau)
        if act.sum() < 10:
            c.append((tau, act.mean(), np.nan)); continue
        pred = np.zeros_like(r['y'])
        pred[r['p_cal'] >= tau] = 1
        c.append((tau, act.mean(), (pred[act] == r['y'][act]).mean()))
    curves[tk] = np.array(c)

# ---------------- Equity curves -------------------------------------
equity   = {}
baseline = {}
for tk, r in results.items():
    long_mask = r['p_cal'] >= TAU_DEPLOY
    daily     = np.where(long_mask, r['ret'], 0.0)
    equity[tk]   = (np.cumprod(1 + daily), long_mask.mean())
    baseline[tk] = np.cumprod(1 + r['ret'])

# ==================================================================
# Plots
# ==================================================================
print("\n[6/6] Plotting...")
fig = plt.figure(figsize=(20, 16))
gs  = fig.add_gridspec(4, 3, hspace=0.40, wspace=0.30)

# 1. Training loss
ax = fig.add_subplot(gs[0, 0])
ax.plot(history['train_loss'], label='Train', color='tab:blue')
ax.plot(history['val_loss'],   label='Val',   color='tab:red')
ax.set(xlabel='Epoch', ylabel='Loss', title='Training & Validation Loss')
ax.legend(); ax.grid(alpha=0.3)

# 2. Val AUC / acc
ax = fig.add_subplot(gs[0, 1])
ax.plot(history['val_acc'], label='Accuracy', color='tab:green')
ax.plot(history['val_auc'], label='AUC',      color='tab:orange')
ax.axhline(0.5, ls='--', color='gray', alpha=0.5)
ax.set(xlabel='Epoch', ylabel='Metric',
       title='Validation Accuracy and AUC')
ax.legend(); ax.grid(alpha=0.3)

# 3. Reliability diagram
ax = fig.add_subplot(gs[0, 2])
bins = np.linspace(0, 1, 11)

def rel_diag(p, y):
    idx = np.digitize(p, bins) - 1
    xs, ys = [], []
    for b in range(10):
        m = idx == b
        if m.sum() > 0:
            xs.append(p[m].mean()); ys.append(y[m].mean())
    return np.array(xs), np.array(ys)

xr, yr = rel_diag(p_val_raw, val_y)
xc, yc = rel_diag(p_val_cal, val_y)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
ax.plot(xr, yr, 'o-', color='tab:red',  label='Raw')
ax.plot(xc, yc, 's-', color='tab:blue', label='Calibrated')
ax.set(xlabel='Predicted prob.', ylabel='Empirical freq.',
       title='Reliability Diagram (Validation)')
ax.legend(); ax.grid(alpha=0.3)

# 4. Per-asset accuracy / AUC
ax = fig.add_subplot(gs[1, 0])
tk_list = list(results.keys())
x = np.arange(len(tk_list)); w = 0.35
ax.bar(x - w/2, [results[t]['acc'] for t in tk_list], w,
       label='Accuracy', color='tab:blue')
ax.bar(x + w/2, [results[t]['auc'] for t in tk_list], w,
       label='AUC',      color='tab:orange')
ax.axhline(0.5, ls='--', color='gray', alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(tk_list)
ax.set(ylabel='Score', title='Per-Asset Accuracy and AUC (Zero-Shot)')
ax.legend(); ax.grid(alpha=0.3)

# 5. Coverage vs selective accuracy
ax = fig.add_subplot(gs[1, 1])
for tk in tk_list:
    c = curves[tk]
    ax.plot(c[:, 1], c[:, 2], 'o-', label=tk, alpha=0.7)
ax.axhline(0.5, ls='--', color='gray', alpha=0.5)
ax.set(xlabel='Coverage', ylabel='Selective accuracy',
       title='Coverage vs Selective Accuracy')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 6. Coverage & selective acc at fixed tau
ax = fig.add_subplot(gs[1, 2])
idx = int(np.argmin(np.abs(taus - TAU_DEPLOY)))
covs = [curves[t][idx, 1] for t in tk_list]
sacc = [curves[t][idx, 2] for t in tk_list]
x = np.arange(len(tk_list))
ax.bar(x - 0.2, covs, 0.4, label=f'Coverage',     color='tab:purple')
ax.bar(x + 0.2, sacc, 0.4, label='Selective acc', color='tab:green')
ax.set_xticks(x); ax.set_xticklabels(tk_list)
ax.set_ylim(0, 1.05)
ax.set(title=f'Coverage and Selective Accuracy at τ={TAU_DEPLOY:.2f}')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 7. Equity curves
ax = fig.add_subplot(gs[2, 0])
for tk in tk_list:
    cum, _ = equity[tk]
    ax.plot(cum, label=f'{tk} strategy', alpha=0.85)
    ax.plot(baseline[tk], '--', alpha=0.35)
ax.set(xlabel='Test days', ylabel='Cumulative return',
       title=f'Equity curves (long if p_cal ≥ {TAU_DEPLOY:.2f}); '
             f'solid=strategy, dashed=buy&hold')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# 8. Prediction distributions
ax = fig.add_subplot(gs[2, 1])
for tk in tk_list[:3]:
    r = results[tk]
    ax.hist(r['p_cal'][r['y'] == 1], bins=25, alpha=0.45,
            density=True, label=f'{tk} y=1')
    ax.hist(r['p_cal'][r['y'] == 0], bins=25, alpha=0.45,
            density=True, label=f'{tk} y=0')
ax.axvline(0.5, ls='--', color='k')
ax.set(xlabel='Calibrated probability', ylabel='Density',
       title='Prediction distributions (first 3 assets)')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# 9. Abstention rate
ax = fig.add_subplot(gs[2, 2])
abst = [1 - curves[t][idx, 1] for t in tk_list]
ax.bar(tk_list, abst, color='tab:red', alpha=0.75)
ax.set(ylabel='Abstention rate',
       title=f'Abstention rate at τ={TAU_DEPLOY:.2f}')
ax.grid(alpha=0.3)

# 10. Summary table
ax = fig.add_subplot(gs[3, :]); ax.axis('off')
rows = []
for tk in tk_list:
    r = results[tk]
    cov  = curves[tk][idx, 1]
    sacc = curves[tk][idx, 2]
    rows.append([tk, r['n'], f"{r['acc']:.4f}", f"{r['auc']:.4f}",
                 f"{r['brier']:.4f}", f"{cov:.3f}",
                 f"{sacc:.4f}" if not np.isnan(sacc) else "—"])
cols = ['Ticker','N','Acc','AUC','Brier','Coverage','Sel.Acc']
tbl = ax.table(cellText=rows, colLabels=cols,
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 1.6)
ax.set_title(f'Test summary (τ={TAU_DEPLOY:.2f})', pad=20)

plt.suptitle(
    f'Dual Head TLSTM — trained on {TRAIN_TICKER}, '
    f'zero-shot on {", ".join(TEST_TICKERS)}',
    fontsize=14, y=0.995)
plt.savefig('dual_head_tlstm_results.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved dual_head_tlstm_results.png")

# ---------------- CSV summary --------------------------------------
summary = pd.DataFrame({
    'ticker':  tk_list,
    'n':       [results[t]['n']     for t in tk_list],
    'accuracy':[results[t]['acc']   for t in tk_list],
    'auc':     [results[t]['auc']   for t in tk_list],
    'brier':   [results[t]['brier'] for t in tk_list],
    'coverage':[curves[t][idx,1]    for t in tk_list],
    'selective_accuracy':[curves[t][idx,2] for t in tk_list],
})
summary.to_csv('dual_head_tlstm_summary.csv', index=False)
print("Saved dual_head_tlstm_summary.csv")