import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, mean_squared_error, mean_absolute_error, r2_score

try:
    import xgboost as xgb
    XGB_OK = True
except Exception:
    XGB_OK = False

try:
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_OK = True
except Exception:
    TORCH_OK = False

def xgb_available(): return XGB_OK
def torch_available(): return TORCH_OK

def _venue_feats(df, venue):
    mid = f'mid_{venue}'; dep = f'depth_{venue}'
    feats = [c for c in [mid, f'{mid}_ret_1', f'{mid}_ret_3', f'{mid}_volatility_10', f'{mid}_diff_to_agg', dep, f'depth_imbalance_{venue}'] if c in df.columns]
    return feats

def prepare_cls_data(df, venue):
    feats = _venue_feats(df, venue)
    dd = df.dropna(subset=feats + ['target_dir']).copy()
    if dd.empty: return None, None, feats
    X = dd[feats].values; y = dd['target_dir'].values
    return X, y, feats

def eval_cls_venue(df, venue):
    out = {'venue': venue, 'auc_mean': None, 'f1_mean': None, 'acc_mean': None, 'valid_splits': 0}
    pack = prepare_cls_data(df, venue)
    if pack[0] is None: return out
    X, y, feats = pack
    tscv = TimeSeriesSplit(n_splits=3)
    aucs, f1s, accs = [], [], []
    scaler = StandardScaler()
    for tr, te in tscv.split(X):
        ytr, yte = y[tr], y[te]
        if len(np.unique(ytr))<2 or len(np.unique(yte))<2: 
            continue
        Xtr, Xte = scaler.fit_transform(X[tr]), scaler.transform(X[te])
        clf = LogisticRegression(max_iter=500).fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:,1]; yhat = clf.predict(Xte)
        aucs.append(roc_auc_score(yte, p)); f1s.append(f1_score(yte, yhat, zero_division=0)); accs.append(accuracy_score(yte, yhat))
    if len(aucs):
        out.update({'auc_mean': float(np.mean(aucs)), 'f1_mean': float(np.mean(f1s)), 'acc_mean': float(np.mean(accs)), 'valid_splits': len(aucs)})
    return out

def prepare_reg_data(df, venue):
    feats = _venue_feats(df, venue)
    dd = df.dropna(subset=feats + ['target_delta']).copy()
    if dd.empty: return None, None, feats
    X = dd[feats].values; y = dd['target_delta'].values
    return X, y, feats

def eval_reg_venue(df, venue):
    out = {'venue': venue, 'mse': None, 'mae': None, 'r2': None, 'valid_splits': 0}
    pack = prepare_reg_data(df, venue)
    if pack[0] is None: return out
    X, y, feats = pack
    tscv = TimeSeriesSplit(n_splits=3)
    mses, maes, r2s = [], [], []
    scaler = StandardScaler()
    for tr, te in tscv.split(X):
        Xtr, Xte = scaler.fit_transform(X[tr]), scaler.transform(X[te])
        ytr, yte = y[tr], y[te]
        if np.isclose(np.std(yte), 0): continue
        reg = Ridge(alpha=1.0).fit(Xtr, ytr)
        pred = reg.predict(Xte)
        mses.append(mean_squared_error(yte, pred)); maes.append(mean_absolute_error(yte, pred)); r2s.append(r2_score(yte, pred))
    if len(mses):
        out.update({'mse': float(np.mean(mses)), 'mae': float(np.mean(maes)), 'r2': float(np.mean(r2s)), 'valid_splits': len(mses)})
    return out

def eval_latency_buckets(raw_df, labeled_df, venue):
    rd = raw_df.copy()
    rd['recv_time'] = pd.to_datetime(rd['recv_time'])
    lat_map = rd[rd['venue']==venue][['recv_time','send_time']].sort_values('recv_time')
    dd = labeled_df.copy()
    dd['grid_time'] = pd.to_datetime(dd['grid_time'])
    merged = pd.merge_asof(dd.sort_values('grid_time'), lat_map, left_on='grid_time', right_on='recv_time', direction='backward')
    merged['latency_s'] = (merged['recv_time'] - merged['send_time']).dt.total_seconds()
    q = merged['latency_s'].quantile([0.25,0.5,0.75]).values
    buckets = {
        'low': merged[merged['latency_s']<=q[0]],
        'mid': merged[(merged['latency_s']>q[0]) & (merged['latency_s']<=q[1])],
        'high': merged[(merged['latency_s']>q[1]) & (merged['latency_s']<=q[2])],
        'tail': merged[merged['latency_s']>q[2]]
    }
    def quick_auc(df_b):
        pack = prepare_cls_data(df_b, venue)
        if pack[0] is None: return None, len(df_b)
        X, y, feats = pack
        if len(np.unique(y))<2: return None, len(df_b)
        Xs = StandardScaler().fit_transform(X)
        clf = LogisticRegression(max_iter=500).fit(Xs, y)
        p = clf.predict_proba(Xs)[:,1]
        return float(roc_auc_score(y, p)), len(df_b)
    res = {b: {'auc': quick_auc(bdf)[0], 'count': quick_auc(bdf)[1]} for b, bdf in buckets.items()}
    return res

# Optional XGBoost
def eval_xgb_cls_venue(df, venue):
    if not XGB_OK:
        return {'venue': venue, 'note': 'xgboost not available'}
    pack = prepare_cls_data(df, venue)
    if pack[0] is None: return {'venue': venue, 'note': 'no data'}
    X, y, feats = pack
    tscv = TimeSeriesSplit(n_splits=3); aucs = []
    for tr, te in tscv.split(X):
        ytr, yte = y[tr], y[te]
        if len(np.unique(ytr))<2 or len(np.unique(yte))<2: continue
        dtr = xgb.DMatrix(X[tr], label=ytr); dte = xgb.DMatrix(X[te], label=yte)
        params = {'objective':'binary:logistic','eval_metric':'auc','eta':0.1,'max_depth':4,'subsample':0.8,'colsample_bytree':0.8}
        bst = xgb.train(params, dtr, num_boost_round=200)
        p = bst.predict(dte)
        aucs.append(roc_auc_score(yte, p))
    return {'venue': venue, 'auc_mean': float(np.mean(aucs)) if aucs else None, 'valid_splits': len(aucs)}

def eval_xgb_reg_venue(df, venue):
    if not XGB_OK:
        return {'venue': venue, 'note': 'xgboost not available'}
    pack = prepare_reg_data(df, venue)
    if pack[0] is None: return {'venue': venue, 'note': 'no data'}
    X, y, feats = pack
    tscv = TimeSeriesSplit(n_splits=3); mses = []
    for tr, te in tscv.split(X):
        dtr = xgb.DMatrix(X[tr], label=y[tr]); dte = xgb.DMatrix(X[te], label=y[te])
        params = {'objective':'reg:squarederror','eval_metric':'rmse','eta':0.1,'max_depth':4,'subsample':0.8,'colsample_bytree':0.8}
        bst = xgb.train(params, dtr, num_boost_round=200)
        pred = bst.predict(dte)
        mses.append(mean_squared_error(y[te], pred))
    return {'venue': venue, 'mse_mean': float(np.mean(mses)) if mses else None, 'valid_splits': len(mses)}

# Optional Tiny Transformer
if TORCH_OK:
    class TinyTransformerReg(nn.Module):
        def __init__(self, d_in, d_model=32, nhead=4, nlayers=2):
            super().__init__()
            self.input = nn.Linear(d_in, d_model)
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True)
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
            self.head = nn.Linear(d_model, 1)
        def forward(self, x):
            z = self.input(x)
            z = self.enc(z)
            y = self.head(z[:, -1, :])
            return y

def eval_torch_reg_venue(df, venue, epochs=30):
    if not TORCH_OK:
        return {'venue': venue, 'note': 'torch not available'}
    Xy = prepare_reg_data(df, venue)
    if Xy[0] is None: return {'venue': venue, 'note': 'no data'}
    X, y, feats = Xy
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    X = X.astype('float32'); y = y.astype('float32')
    n = len(y); split = int(n*0.8)
    Xtr, Xte = X[:split], X[split:]; ytr, yte = y[:split], y[split:]
    train_ds = TensorDataset(torch.from_numpy(Xtr).unsqueeze(1), torch.from_numpy(ytr).unsqueeze(1))
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    model = TinyTransformerReg(d_in=X.shape[1]) if TORCH_OK else None
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        for xb, yb in train_dl:
            opt.zero_grad(); pred = model(xb); loss = loss_fn(pred, yb); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        te_pred = model(torch.from_numpy(Xte).unsqueeze(1)).squeeze(1).numpy()
    mse = float(((te_pred - yte)**2).mean())
    return {'venue': venue, 'mse': mse, 'epochs': epochs}
