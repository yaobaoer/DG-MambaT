# -*- coding: utf-8 -*-
"""
Leave-One-Device-Out (LODO) — IoT fusion weather (7 devices)
训练: 清零留出设备 d 的特征（模型从未见过该设备通道）
测试: 7 路全开官方 TEST（d 在测试时首次出现）
对齐目录 1 跨协议 LOPO：Target mean±std / Gap / 四项指标
"""
import os, time, warnings, re, tempfile, json
warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torch_geometric.nn import GATv2Conv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch

_ROOT = Path(r'E:\apt\YES\2')
os.chdir(_ROOT)
_TMP = _ROOT / '_tmp'
_TMP.mkdir(exist_ok=True)
for _k in ('TEMP', 'TMP', 'TMPDIR'):
    os.environ[_k] = str(_TMP)
tempfile.tempdir = str(_TMP)

DATA_CSV = _ROOT / 'iot_fusion_weather_all.csv'
OUT_TXT = _ROOT / 'lodo_leave_one_device_out.txt'
OUT_CSV = _ROOT / 'lodo_leave_one_device_out.csv'
OUT_METRICS_CSV = _ROOT / 'lodo_leave_one_device_metrics.csv'
CACHE_DIR = _TMP / 'lodo_cache'
CACHE_DIR.mkdir(exist_ok=True)
MODEL_DIR = _ROOT / 'mmm'

DEVICES = ['weather', 'modbus', 'thermostat', 'fridge', 'garage_door', 'gps', 'motion_light']
WINDOW, TRAIN_STEP, EVAL_STEP = 128, 2, 2
BATCH, EPOCHS, LR, PATIENCE = 64, 6, 1e-3, 2
BATCH_G = 24
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SOURCE_FILES = {
    'Ours': 'best_gat_mamba_ultralite_results.txt',
    'CKAN-BiLSTM': 'best_ckan_bilstm_results.txt',
    'GRID': 'best_grid_results.txt',
    'Transformer-IDS': 'best_tb_results.txt',
    'WaveMamba': 'best_wavemamba_results.txt',
    'FeCo': 'best_feco_results.txt',
    'MPGNN': 'best_mpgnn_results.txt',
    'TCG-IDS': 'best_tcg_ids_results.txt',
}

MODEL_NAMES = [
    'Ours',
    'CKAN-BiLSTM', 'WaveMamba', 'Transformer-IDS', 'FeCo',
    'GRID', 'MPGNN', 'TCG-IDS',
]


def parse_test_f1(results_path):
    if not results_path.is_file():
        return float('nan')
    for line in results_path.read_text(encoding='utf-8', errors='ignore').splitlines():
        m = re.search(r'test_macro_f1\s*=\s*([0-9.]+)', line, re.I)
        if m:
            return float(m.group(1))
    return float('nan')


def stratified_timegroup_split(df):
    df = df.copy()
    df['time_group'] = df.index // 2000
    tr, va, te = [], [], []
    for _, g in df.groupby('time_group'):
        if len(g) < 3:
            tr.extend(g.index.tolist())
            continue
        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
            tv, tes = next(sss.split(g, g['label_encoded']))
        except ValueError:
            tv, tes = np.arange(len(g)), np.array([], dtype=int)
        if len(tv) > 1:
            try:
                sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
                sub = g.iloc[tv]
                a, b = next(sss2.split(sub, sub['label_encoded']))
                tr.extend(g.index[tv][a].tolist())
                va.extend(g.index[tv][b].tolist())
            except ValueError:
                tr.extend(g.index[tv].tolist())
        else:
            tr.extend(g.index[tv].tolist())
        if len(tes):
            te.extend(g.index[tes].tolist())
    return (df.loc[tr].sort_index().reset_index(drop=True),
            df.loc[va].sort_index().reset_index(drop=True),
            df.loc[te].sort_index().reset_index(drop=True))


class FusionWindows(Dataset):
    def __init__(self, df, window, step, scaler=None, fit=False):
        self.device_dims, cols = [], []
        for d in DEVICES:
            c = [x for x in df.columns if x.startswith(d + '_')
                 and not x.endswith('_label') and not x.endswith('_type')]
            self.device_dims.append(len(c))
            cols.extend(c)
        X = df[cols].apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
        y = df['label_encoded'].values.astype(np.int64)
        if fit:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler
            X = self.scaler.transform(X)
        X = np.nan_to_num(np.clip(X, -8, 8), nan=0.0).astype(np.float32)
        xs, ys = [], []
        for i in range(0, len(X) - window + 1, step):
            xs.append(X[i:i + window])
            ys.append(int(y[i + window - 1]))
        self.X = torch.from_numpy(np.stack(xs).astype(np.float32))
        self.y = torch.from_numpy(np.asarray(ys, np.int64))
        sl, s = [], 0
        for d in self.device_dims:
            sl.append((s, s + d))
            s += d
        self.slices = sl
        print(f'windows={len(self)} dims={self.device_dims} step={step}')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def mask_devices(x, slices, idxs):
    if idxs is None:
        return x
    if isinstance(idxs, int):
        idxs = [idxs]
    x = x.clone()
    for idx in idxs:
        a, b = slices[idx]
        x[:, :, a:b] = 0
    return x


class SimpleMamba(nn.Module):
    def __init__(self, d_model, d_hidden=32, kernel=3):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_hidden, kernel, padding=kernel - 1)
        self.gate = nn.Conv1d(d_model, d_hidden, kernel, padding=kernel - 1)
        self.out = nn.Linear(d_hidden, d_model)
        self.n = nn.LayerNorm(d_model)

    def forward(self, x):
        xt, L = x.transpose(1, 2), x.size(1)
        h = (self.conv(xt)[:, :, :L] * torch.sigmoid(self.gate(xt)[:, :, :L])).transpose(1, 2)
        return self.n(self.out(h) + x)


class UltraLite(nn.Module):
    def __init__(self, device_dims, n_cls, d_model=16, dropout=0.1):
        super().__init__()
        self.device_dims = device_dims
        self.n = len(device_dims)
        self.d_model = d_model
        self.device_projectors = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d_model), nn.LayerNorm(d_model), nn.GELU())
            for d in device_dims
        ])
        self.share_logit = nn.Parameter(torch.tensor(-2.0))
        self.act_temp = nn.Parameter(torch.tensor(1.0))
        pw = torch.full((self.n, self.n), 0.1)
        for i, j in [(4, 5), (5, 4), (2, 3), (3, 2), (4, 6), (6, 4)]:
            pw[i, j] = 1.0
        for i, j in [(0, 1), (1, 0), (0, 2), (2, 0)]:
            pw[i, j] = 0.2
        self.register_buffer('prior', pw)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(),
                                  nn.Linear(d_model, 1), nn.Sigmoid())
        self.gat = GATv2Conv(d_model, d_model, heads=1, concat=False, dropout=dropout,
                             add_self_loops=False, edge_dim=1)
        self.n1 = nn.LayerNorm(d_model)
        md = d_model * self.n
        self.mamba = SimpleMamba(md, 32, 3)
        self.attn = nn.Linear(md, 1)
        self.clf = nn.Sequential(nn.LayerNorm(md * 2), nn.Linear(md * 2, n_cls))

    def forward(self, x):
        B, T, _ = x.shape
        feats, s = [], 0
        for proj, d in zip(self.device_projectors, self.device_dims):
            feats.append(proj(x[:, :, s:s + d]))
            s += d
        h = torch.stack(feats, 2)
        act = h.norm(dim=-1, keepdim=True)
        w = torch.softmax(act / self.act_temp.clamp(0.2, 5), dim=2)
        shared = (h * w).sum(2, keepdim=True)
        g = torch.sigmoid(self.share_logit)
        h = (1 - g) * h + g * shared
        pooled = h.mean(1)
        N, D = self.n, self.d_model
        ei = torch.tensor([[i, j] for i in range(N) for j in range(N) if i != j],
                          dtype=torch.long, device=x.device).t().contiguous()
        nm = F.normalize(pooled, dim=-1)
        sim = nm @ nm.transpose(1, 2)
        dyn = torch.stack([sim[:, i, j] for i, j in zip(ei[0], ei[1])], -1)
        fi, fj = pooled[:, ei[0]], pooled[:, ei[1]]
        al = self.gate(torch.cat([fi, fj], -1)).squeeze(-1)
        pr = self.prior[ei[0], ei[1]].view(1, -1)
        ew = (al * pr + (1 - al) * dyn).reshape(-1, 1)
        off = torch.arange(B, device=x.device) * N
        eib = (ei.unsqueeze(1) + off.view(1, -1, 1)).reshape(2, -1)
        u = self.n1(pooled.reshape(-1, D) + self.gat(pooled.reshape(-1, D), eib, edge_attr=ew))
        fused = self.mamba((h + u.reshape(B, 1, N, D)).reshape(B, T, -1))
        aw = torch.softmax(self.attn(fused).squeeze(-1), 1)
        return self.clf(torch.cat([(fused * aw.unsqueeze(-1)).sum(1), fused.mean(1)], -1))


class CKANBiLSTM(nn.Module):
    def __init__(self, in_dim, n_cls):
        super().__init__()
        self.c1 = nn.Conv1d(in_dim, 32, 3, padding=1)
        self.c2 = nn.Conv1d(32, 64, 3, padding=1)
        self.n1 = nn.InstanceNorm1d(32, affine=True)
        self.n2 = nn.InstanceNorm1d(64, affine=True)
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_cls))

    def forward(self, x):
        h = F.relu(self.n1(self.c1(x.transpose(1, 2))))
        h = F.relu(self.n2(self.c2(h))).transpose(1, 2)
        h, _ = self.lstm(h)
        return self.fc(h.mean(1))


class CGATN(nn.Module):
    def __init__(self, in_dim, n_cls, n_layers=2, hidden=64, heads=4):
        super().__init__()
        ph = hidden // heads
        self.gats = nn.ModuleList()
        for i in range(n_layers):
            self.gats.append(GATConv(in_dim if i == 0 else hidden, ph, heads=heads, concat=True))
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 8))
                                   for _ in range(n_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(8) for _ in range(n_layers)])
        self.fc = nn.Sequential(nn.Linear(n_layers * 8, 64), nn.ReLU(), nn.Linear(64, n_cls))

    def forward(self, data):
        h, ei, batch = data.x, data.edge_index, data.batch
        zs = []
        for gat, mlp, bn in zip(self.gats, self.mlps, self.bns):
            h = F.relu(gat(h, ei))
            zs.append(bn(mlp(global_mean_pool(h, batch))))
        return self.fc(torch.cat(zs, -1))


class TransformerIDS(nn.Module):
    def __init__(self, in_dim, n_cls, T=128, d_model=28, nhead=4, n_layers=2):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, T, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, 112, 0.1, batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.Linear(T * d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, n_cls))

    def forward(self, x):
        B, T, _ = x.shape
        z = self.enc(self.proj(x) + self.pos[:, :T])
        return self.head(z.reshape(B, -1))


class TCGWrap(nn.Module):
    def __init__(self, in_dim, n_cls, d_model=64):
        super().__init__()
        self.p = nn.Linear(in_dim, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.clf = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_cls))

    def forward(self, x):
        h, _ = self.gru(F.gelu(self.p(x)))
        return self.clf(h.mean(1))


class WaveLite(nn.Module):
    def __init__(self, in_dim, n_cls, d_model=64):
        super().__init__()
        self.in_p = nn.Sequential(nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.dw = nn.Conv1d(d_model, d_model, 4, padding=3, groups=d_model)
        self.n = nn.LayerNorm(d_model)
        self.out = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_cls))

    def forward(self, x):
        h = self.in_p(x)
        h = h + F.silu(self.dw(self.n(h).transpose(1, 2))[:, :, :x.size(1)].transpose(1, 2))
        return self.out(h.mean(1))


class FeCoClf(nn.Module):
    def __init__(self, in_dim, n_cls):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 64))
        self.head = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, n_cls))

    def forward(self, x):
        return self.head(F.relu(self.enc(x[:, -1])))


class MPGNNLite(nn.Module):
    def __init__(self, in_dim, n_cls, d_model=32):
        super().__init__()
        self.p = nn.Linear(in_dim, d_model)
        self.clf = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_cls))

    def forward(self, x):
        return self.clf(F.gelu(self.p(x)).mean(1))


_CHAIN = None


def to_grid_batch(x):
    global _CHAIN
    B, T, _ = x.shape
    if _CHAIN is None or _CHAIN.size(1) != 2 * (T - 1):
        src = np.arange(T - 1)
        dst = np.arange(1, T)
        _CHAIN = torch.tensor(np.stack([np.r_[src, dst], np.r_[dst, src]]), dtype=torch.long)
    return Batch.from_data_list([Data(x=x[b], edge_index=_CHAIN.to(x.device)) for b in range(B)])


def build_model(name, dims, n_cls):
    in_dim = int(sum(dims))
    return {
        'Ours': lambda: UltraLite(dims, n_cls),
        'CKAN-BiLSTM': lambda: CKANBiLSTM(in_dim, n_cls),
        'GRID': lambda: CGATN(in_dim, n_cls),
        'Transformer-IDS': lambda: TransformerIDS(in_dim, n_cls, T=WINDOW),
        'TCG-IDS': lambda: TCGWrap(in_dim, n_cls),
        'WaveMamba': lambda: WaveLite(in_dim, n_cls),
        'FeCo': lambda: FeCoClf(in_dim, n_cls),
        'MPGNN': lambda: MPGNNLite(in_dim, n_cls),
    }[name]()


def macro_metrics(y, pred, n_cls):
    labels = list(range(n_cls))
    return {
        'acc': float(accuracy_score(y, pred)),
        'prec': float(precision_score(y, pred, average='macro', labels=labels, zero_division=0)),
        'rec': float(recall_score(y, pred, average='macro', labels=labels, zero_division=0)),
        'f1': float(f1_score(y, pred, average='macro', labels=labels, zero_division=0)),
    }


def _fold_metric_val(x, key='f1'):
    if isinstance(x, dict):
        return float(x[key])
    return float(x)


def _cache_load_fold(pack, model_name):
    if f'{model_name}_f1' in pack:
        return {
            'acc': float(pack[f'{model_name}_acc']),
            'prec': float(pack[f'{model_name}_prec']),
            'rec': float(pack[f'{model_name}_rec']),
            'f1': float(pack[f'{model_name}_f1']),
        }
    if model_name in pack:
        return {'f1': float(pack[model_name])}
    return None


def _cache_save_fold(fold_scores):
    flat = {}
    for name, met in fold_scores.items():
        if isinstance(met, dict):
            for k, v in met.items():
                flat[f'{name}_{k}'] = float(v)
        else:
            flat[name] = float(met)
    return flat


def class_weight_from_y(y, n_cls):
    cnt = np.bincount(y, minlength=n_cls).astype(np.float64)
    w = 1.0 / np.maximum(cnt, 1.0)
    w = w / w.sum() * n_cls
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def eval_loader(model, loader, slices, n_cls, hold_d=None, is_grid=False):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if hold_d is not None:
            x = mask_devices(x, slices, hold_d)
        logits = model(to_grid_batch(x)) if is_grid else model(x)
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(y.cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return macro_metrics(y, p, n_cls)


def train_model(model, tr_ld, va_ld, slices, hold_d, n_cls, cw, is_grid=False):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    best_f1, best_state, bad = -1.0, None, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for x, y in tr_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = mask_devices(x, slices, hold_d)
            opt.zero_grad()
            logits = model(to_grid_batch(x)) if is_grid else model(x)
            loss = F.cross_entropy(logits.float(), y, weight=cw, label_smoothing=0.01)
            if torch.isnan(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        vf1 = eval_loader(model, va_ld, slices, n_cls, hold_d=None, is_grid=is_grid)['f1']
        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, best_f1


def _ckpt_path(dname, name):
    safe = name.replace('/', '_').replace('\\', '_')
    return MODEL_DIR / dname / f'{safe}.pth'


def save_run_meta(le, tr_ds):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        'classes': list(le.classes_),
        'device_dims': list(tr_ds.device_dims),
        'slices': [list(s) for s in tr_ds.slices],
        'devices': DEVICES,
        'window': WINDOW,
        'train_step': TRAIN_STEP,
        'eval_step': EVAL_STEP,
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'seed': SEED,
    }
    (MODEL_DIR / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez(MODEL_DIR / 'scaler.npz', mean_=tr_ds.scaler.mean_, scale_=tr_ds.scaler.scale_)
    print(f'saved meta -> {MODEL_DIR / "meta.json"}')


def save_fold_ckpt(path, model, name, dname, hold_idx, metrics, is_grid):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        'model_name': name,
        'hold_device': dname,
        'hold_idx': hold_idx,
        'is_grid': is_grid,
        'metrics': metrics,
        'target_f1': float(metrics['f1']),
        'task': 'LODO',
    }, path)


def load_lodo_model(device_name, model_name='Ours', map_location=None):
    """从 mmm/{device_name}/{model_name}.pth 加载 LODO 折权重（测试用 Full-7）。"""
    ckpt_p = _ckpt_path(device_name, model_name)
    if not ckpt_p.is_file():
        raise FileNotFoundError(f'缺少 {ckpt_p}')
    meta_p = MODEL_DIR / 'meta.json'
    if not meta_p.is_file():
        raise FileNotFoundError(f'缺少 {MODEL_DIR / "meta.json"}，请先跑完 _run_lodo.py')
    meta = json.loads(meta_p.read_text(encoding='utf-8'))
    pack = torch.load(ckpt_p, map_location=map_location or 'cpu', weights_only=False)
    model = build_model(model_name, meta['device_dims'], len(meta['classes']))
    model.load_state_dict(pack['model_state'])
    return model, pack, meta


def run_one_model(name, dname, hold_idx, tr_ld, va_ld, te_ld, slices, n_cls, cw, dims):
    is_grid = name == 'GRID'
    ckpt_p = _ckpt_path(dname, name)
    if ckpt_p.is_file():
        pack = torch.load(ckpt_p, map_location=DEVICE, weights_only=False)
        model = build_model(name, dims, n_cls)
        model.load_state_dict(pack['model_state'])
        model = model.to(DEVICE)
        sc = eval_loader(model, te_ld, slices, n_cls, hold_d=None, is_grid=is_grid)
        print(f'  {name} Target-F1={sc["f1"]:.4f} [loaded mmm]')
        del model
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        return sc
    t0 = time.time()
    model = build_model(name, dims, n_cls)
    model, _ = train_model(model, tr_ld, va_ld, slices, hold_d=hold_idx, n_cls=n_cls, cw=cw, is_grid=is_grid)
    sc = eval_loader(model, te_ld, slices, n_cls, hold_d=None, is_grid=is_grid)
    save_fold_ckpt(ckpt_p, model, name, dname, hold_idx, sc, is_grid)
    print(f'  {name} Target-F1={sc["f1"]:.4f} ({time.time() - t0:.0f}s) -> {ckpt_p}')
    del model
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return sc


METRIC_KEYS = [('acc', 'Accuracy'), ('prec', 'Precision'), ('rec', 'Recall'), ('f1', 'F1')]


def _agg_fold(results, model, devs, key):
    vals = np.array([_fold_metric_val(results[model][p], key) for p in devs], dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=1))


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df = pd.read_csv(DATA_CSV, low_memory=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    valid = df['final_type'].value_counts()[df['final_type'].value_counts() > 10].index
    df = df[df['final_type'].isin(valid)].copy()
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['final_type'])
    n_cls = len(le.classes_)
    print('classes', list(le.classes_))

    df_tr, df_va, df_te = stratified_timegroup_split(df)
    print(f'rows train/val/test {len(df_tr)}/{len(df_va)}/{len(df_te)}')

    tr_ds = FusionWindows(df_tr, WINDOW, TRAIN_STEP, fit=True)
    va_ds = FusionWindows(df_va, WINDOW, EVAL_STEP, scaler=tr_ds.scaler)
    te_ds = FusionWindows(df_te, WINDOW, EVAL_STEP, scaler=tr_ds.scaler)
    slices = tr_ds.slices
    counts = np.maximum(np.bincount(tr_ds.y.numpy(), minlength=n_cls).astype(np.float64), 1)
    sw = 1.0 / np.sqrt(counts[tr_ds.y.numpy()])
    sw = sw / sw.mean()
    tr_ld = DataLoader(tr_ds, BATCH, sampler=WeightedRandomSampler(torch.DoubleTensor(sw), len(tr_ds), True))
    va_ld = DataLoader(va_ds, BATCH, shuffle=False)
    te_ld = DataLoader(te_ds, BATCH, shuffle=False)
    cw = torch.tensor(np.clip(np.sqrt(counts.sum() / (n_cls * counts)), 0.5, 3.0),
                      dtype=torch.float32, device=DEVICE)

    save_run_meta(le, tr_ds)

    SOURCE = {m: parse_test_f1(_ROOT / fn) for m, fn in SOURCE_FILES.items()}
    print('Source F1 (official full-7 ckpt):', {k: round(v, 4) for k, v in SOURCE.items()})

    results = {m: {} for m in MODEL_NAMES}
    dims = tr_ds.device_dims

    for d, dname in enumerate(DEVICES):
        cache_p = CACHE_DIR / f'fold_{dname}.npz'
        all_ckpt = all(_ckpt_path(dname, m).is_file() for m in MODEL_NAMES)

        if all_ckpt:
            print(f'\n[mmm] fold={dname} all ckpts found, eval only')
        else:
            print(f'\n######## LODO fold hold-out device = {dname} ########')

        fold_scores = {}
        for name in MODEL_NAMES:
            fold_scores[name] = run_one_model(
                name, dname, d, tr_ld, va_ld, te_ld, slices, n_cls, cw, dims)

        np.savez(cache_p, **_cache_save_fold(fold_scores))
        for m, sc in fold_scores.items():
            results[m][dname] = sc

    rows, metric_rows = [], []
    for m in MODEL_NAMES:
        f1_scores = np.array([_fold_metric_val(results[m][p], 'f1') for p in DEVICES], dtype=np.float64)
        src = SOURCE.get(m, float('nan'))
        rows.append({
            'model': m,
            'Source_F1': src,
            'Target_mean': float(f1_scores.mean()),
            'Target_std': float(f1_scores.std(ddof=1)),
            'Target_worst': float(f1_scores.min()),
            'Gap_mean': float(src - f1_scores.mean()) if np.isfinite(src) else float('nan'),
            'Gap_worst': float(src - f1_scores.min()) if np.isfinite(src) else float('nan'),
            **{f'T_{p}': _fold_metric_val(results[m][p], 'f1') for p in DEVICES},
        })
        mrow = {'model': m}
        for k, label in METRIC_KEYS:
            mu, sd = _agg_fold(results, m, DEVICES, k)
            mrow[f'{label}_mean'] = mu
            mrow[f'{label}_std'] = sd
        metric_rows.append(mrow)

    df_sum = pd.DataFrame(rows).set_index('model')
    df_sum['score'] = df_sum['Target_mean'] - 0.25 * df_sum['Gap_mean'].fillna(0)
    show = df_sum.sort_values('score', ascending=False)
    cols = ['Source_F1', 'Target_mean', 'Target_std', 'Target_worst', 'Gap_mean', 'Gap_worst']

    print('\n' + '=' * 88)
    print('LODO: train w/o device-d features → test full-7 features')
    print(show[cols].round(4).to_string())
    print('=' * 88)
    print('\nPer-device Target F1:')
    print(show[[f'T_{p}' for p in DEVICES]].round(4).to_string())

    ours = df_sum.loc['Ours']
    print('\nOurs - others:')
    for m, row in df_sum.iterrows():
        if m == 'Ours':
            continue
        print(f'  vs {m:20s}  dMean={ours.Target_mean - row.Target_mean:+.4f}  '
              f'dWorst={ours.Target_worst - row.Target_worst:+.4f}  '
              f'dGap={row.Gap_mean - ours.Gap_mean:+.4f}')

    lines = [
        'Leave-One-Device-Out (LODO) on IoT fusion weather',
        'Train: zero held-out device features | Test: full 7-device features',
        f'window={WINDOW} step={TRAIN_STEP}/{EVAL_STEP} epochs={EPOCHS} patience={PATIENCE} seed={SEED}',
        '',
        show[cols].to_string(float_format=lambda x: f'{x:.6f}'),
        '',
        'Per-device Target F1:',
        show[[f'T_{p}' for p in DEVICES]].to_string(float_format=lambda x: f'{x:.6f}'),
        '',
        'Delta Ours - other:',
    ]
    for m, row in df_sum.iterrows():
        if m == 'Ours':
            continue
        lines.append(
            f'  vs {m:20s}  dMean={ours.Target_mean - row.Target_mean:+.6f}  '
            f'dWorst={ours.Target_worst - row.Target_worst:+.6f}  '
            f'dGap(flip)={row.Gap_mean - ours.Gap_mean:+.6f}')

    OUT_TXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    show.to_csv(OUT_CSV)
    df_metrics = pd.DataFrame(metric_rows).set_index('model')
    df_metrics.sort_values('F1_mean', ascending=False).to_csv(OUT_METRICS_CSV)
    print(f'\nsaved {OUT_TXT}')
    print(f'saved {OUT_CSV}')
    print(f'saved {OUT_METRICS_CSV}')
    print(f'checkpoints -> {MODEL_DIR}')


if __name__ == '__main__':
    main()
