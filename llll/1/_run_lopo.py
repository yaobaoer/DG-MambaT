# -*- coding: utf-8 -*-
"""
Leave-One-Protocol-Out (LOPO) — X-IIoTID fusion
协议: 训练时清零协议 p 的特征（模型从未见过该通道）；
      测试用全特征官方 TEST（p 在测试时首次出现）。
报: Target Macro-F1 / Gap(=Source-Target) / Worst-protocol / mean±std
对比: Ours(soft-share) vs CKAN/GRID/Wave/TF/FeCo/MPGNN/TCG
"""
import os, sys, time, ast, json, re, warnings, tempfile, copy
warnings.filterwarnings('ignore')
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, GATConv, SAGEConv, global_mean_pool

_ROOT = Path(r'E:\apt\YES\1')
os.chdir(_ROOT)
_TMP = _ROOT / '_tmp'
_TMP.mkdir(exist_ok=True)
for _k in ('TEMP', 'TMP', 'TMPDIR'):
    os.environ[_k] = str(_TMP)
tempfile.tempdir = str(_TMP)

DATA_CSV = _ROOT / 'xiiotid_class1_fusion.csv'
CKPT_OURS = _ROOT / 'best_gat_mamba_ultralite.pth'
OUT_TXT = _ROOT / 'lopo_leave_one_protocol_out.txt'
OUT_CSV = _ROOT / 'lopo_leave_one_protocol_out.csv'
OUT_METRICS_CSV = _ROOT / 'lopo_leave_one_protocol_metrics.csv'
CACHE_DIR = _TMP / 'lopo_cache'
CACHE_DIR.mkdir(exist_ok=True)

WINDOW, TRAIN_STEP, EVAL_STEP = 48, 6, 48
MAX_TRAIN, MAX_EVAL = 16000, 12000
MIN_PURITY = 0.50
BATCH = 64
BATCH_G = 24
EPOCHS = 6
PATIENCE = 2
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROTOCOL_GROUPS = ['arp', 'icmp', 'http', 'tcp', 'udp', 'dns', 'mqtt', 'mbtcp']
META_COLS = ('final_label', 'final_type', 'label_encoded', 'seq_id', 'stream_id', 'split', 'Timestamp')
NUM_CLASSES = 16
SKIP_FUNCS = set()
ROOT = _ROOT
SECOND = _ROOT
SEED = 42


# ===================== data =====================
class EdgeIIoTFusionDataset(Dataset):
    def __init__(self, df, window_size=48, step=6, scaler=None, fit_scaler=False,
                 max_windows=None, min_purity=0.45, normal_idx=None):
        self.device_dims, self.all_feat_cols = [], []
        df = df.copy().reset_index(drop=True)
        uniq = {old: i for i, old in enumerate(sorted(df['stream_id'].unique()))}
        df['stream_id'] = df['stream_id'].map(uniq).astype(np.int64)
        for proto in PROTOCOL_GROUPS:
            cols = [c for c in df.columns
                    if (c == proto or c.startswith(proto + '_')) and c not in META_COLS]
            if not cols:
                cols = [f'{proto}_placeholder']
                df[cols[0]] = np.float32(0.0)
            self.device_dims.append(len(cols))
            self.all_feat_cols.extend(cols)
        X_all = df[self.all_feat_cols].values.astype(np.float32)
        y_all = df['label_encoded'].values.astype(np.int64)
        sid_all = df['stream_id'].values.astype(np.int64)
        if scaler is None and fit_scaler:
            self.scaler = StandardScaler()
            X_all = self.scaler.fit_transform(X_all)
        elif scaler is not None:
            self.scaler = scaler
            X_all = self.scaler.transform(X_all)
        else:
            self.scaler = None
        X_all = np.nan_to_num(np.clip(X_all, -8, 8), nan=0.0).astype(np.float32)
        samples, labels = [], []
        for sid in np.unique(sid_all):
            mask = sid_all == sid
            X, y = X_all[mask], y_all[mask]
            if len(X) < window_size:
                continue
            for i in range(0, len(X) - window_size + 1, step):
                yw = y[i:i + window_size]
                vals, cnts = np.unique(yw, return_counts=True)
                frac = {int(v): float(c) / len(yw) for v, c in zip(vals, cnts)}
                atk = [(f, c) for c, f in frac.items()
                       if normal_idx is None or c != normal_idx]
                if atk and max(atk)[0] >= 0.55:
                    lab, purity = max(atk)[1], max(atk)[0]
                else:
                    top = int(cnts.max())
                    tied = vals[cnts == top]
                    lab = int(tied[0]) if len(tied) == 1 else int(yw[-1])
                    purity = float(top) / len(yw)
                if purity < min_purity:
                    continue
                samples.append(X[i:i + window_size])
                labels.append(lab)
        if max_windows is not None and len(samples) > max_windows:
            rng = np.random.RandomState(0)
            labels_arr = np.asarray(labels)
            classes = np.unique(labels_arr)
            raw = np.array([(labels_arr == c).sum() for c in classes], dtype=np.float64)
            alloc = np.sqrt(raw)
            alloc = alloc / alloc.sum() * max_windows
            floor = max(32, max_windows // (len(classes) * 8))
            alloc = np.maximum(np.floor(alloc), floor).astype(np.int64)
            while alloc.sum() > max_windows:
                donors = np.where(alloc > floor)[0]
                if len(donors) == 0:
                    break
                alloc[donors[np.argmax(alloc[donors])]] -= 1
            keep = []
            for c, n_keep in zip(classes, alloc):
                idx = np.where(labels_arr == c)[0]
                if len(idx) > n_keep:
                    idx = rng.choice(idx, int(n_keep), replace=False)
                keep.extend(idx.tolist())
            if len(keep) > max_windows:
                keep = rng.choice(keep, max_windows, replace=False).tolist()
            samples = [samples[i] for i in keep]
            labels = [labels[i] for i in keep]
        self.samples = np.stack(samples).astype(np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.samples[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


def zero_protocol(X, device_dims, p):
    out = X.copy()
    off = int(np.sum(device_dims[:p]))
    out[:, :, off:off + device_dims[p]] = 0.0
    return out


def zero_protocols(X, device_dims, proto_idxs):
    out = X.copy()
    for p in proto_idxs:
        off = int(np.sum(device_dims[:p]))
        out[:, :, off:off + device_dims[p]] = 0.0
    return out


# ===================== UltraLite =====================
class SimpleMamba(nn.Module):
    def __init__(self, d_model, d_hidden, kernel=4):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_hidden, kernel_size=kernel, padding=kernel - 1)
        self.gate = nn.Conv1d(d_model, d_hidden, kernel_size=kernel, padding=kernel - 1)
        self.out_proj = nn.Linear(d_hidden, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_t, L = x.transpose(1, 2), x.size(1)
        h = (self.conv(x_t)[:, :, :L] * torch.sigmoid(self.gate(x_t)[:, :, :L])).transpose(1, 2)
        return self.norm(self.out_proj(h) + x)


class InnovativeGATWithMambaUltraLite(nn.Module):
    def __init__(self, device_dims, d_model=16, d_hidden=32, num_classes=16,
                 gat_heads=1, dropout=0.1, use_soft_share=True):
        super().__init__()
        self.num_devices = len(device_dims)
        self.device_dims = device_dims
        self.d_model = d_model
        self.use_soft_share = use_soft_share
        self.device_projectors = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, d_model), nn.LayerNorm(d_model), nn.GELU())
            for dim in device_dims
        ])
        if use_soft_share:
            self.share_logit = nn.Parameter(torch.tensor(-2.0))
            self.act_temp = nn.Parameter(torch.tensor(1.0))
        prior_weight = torch.full((self.num_devices, self.num_devices), 0.1)
        for i, j in [(2, 3), (3, 2), (3, 4), (4, 3), (4, 5), (5, 4), (6, 3), (3, 6), (7, 3), (3, 7)]:
            if i < self.num_devices and j < self.num_devices:
                prior_weight[i, j] = 1.0
        for i, j in [(0, 1), (1, 0), (0, 3), (3, 0), (1, 3), (3, 1)]:
            if i < self.num_devices and j < self.num_devices:
                prior_weight[i, j] = 0.2
        self.register_buffer('prior_weight', prior_weight)
        self.dynamic_prior_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, 1), nn.Sigmoid())
        self.gat1 = GATv2Conv(d_model, d_model, heads=gat_heads, concat=False,
                              dropout=dropout, add_self_loops=False, edge_dim=1)
        self.n1 = nn.LayerNorm(d_model)
        mamba_dim = d_model * self.num_devices
        self.mamba = SimpleMamba(mamba_dim, d_hidden, kernel=3)
        self.attn = nn.Linear(mamba_dim, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(mamba_dim * 2), nn.Linear(mamba_dim * 2, num_classes))

    def calibrate_cross_protocol(self, device_feats):
        act = device_feats.norm(dim=-1, keepdim=True)
        temp = self.act_temp.clamp(0.2, 5.0)
        w = torch.softmax(act / temp, dim=2)
        shared = (device_feats * w).sum(dim=2, keepdim=True)
        gamma = torch.sigmoid(self.share_logit)
        return (1.0 - gamma) * device_feats + gamma * shared

    def compute_dynamic_edges(self, pooled):
        B, N, D = pooled.shape
        device = pooled.device
        ei = torch.tensor(
            [[i, j] for i in range(N) for j in range(N) if i != j],
            dtype=torch.long, device=device).t().contiguous()
        normed = F.normalize(pooled, p=2, dim=-1)
        sim = torch.matmul(normed, normed.transpose(1, 2))
        dyn = torch.stack([sim[:, i, j] for i, j in zip(ei[0], ei[1])], dim=-1)
        fi, fj = pooled[:, ei[0]], pooled[:, ei[1]]
        alpha = self.dynamic_prior_gate(torch.cat([fi, fj], -1)).squeeze(-1)
        prior = self.prior_weight[ei[0], ei[1]].view(1, -1)
        if self.use_soft_share:
            agree = ((fi * fj).sum(-1) / (fi.norm(dim=-1) * fj.norm(dim=-1) + 1e-6)).clamp(0, 1)
            ew = ((alpha * prior + (1 - alpha) * dyn) * (0.5 + 0.5 * agree)).reshape(-1, 1)
        else:
            ew = (alpha * prior + (1 - alpha) * dyn).reshape(-1, 1)
        off = torch.arange(B, device=device) * N
        eib = (ei.unsqueeze(1) + off.view(1, -1, 1)).reshape(2, -1)
        return eib, ew

    def forward(self, x):
        B, T, _ = x.shape
        device_feats, split_idx = [], 0
        for proj, dim in zip(self.device_projectors, self.device_dims):
            device_feats.append(proj(x[:, :, split_idx:split_idx + dim]))
            split_idx += dim
        device_feats = torch.stack(device_feats, dim=2)
        if self.use_soft_share:
            device_feats = self.calibrate_cross_protocol(device_feats)
        pooled = device_feats.mean(1)
        edge_index, edge_weights = self.compute_dynamic_edges(pooled)
        h = pooled.reshape(-1, self.d_model)
        h = self.n1(h + self.gat1(h, edge_index, edge_attr=edge_weights))
        out_seq = device_feats + h.reshape(B, 1, self.num_devices, self.d_model)
        fused = self.mamba(out_seq.reshape(B, T, -1))
        w = torch.softmax(self.attn(fused).squeeze(-1), dim=1)
        feat = torch.cat([(fused * w.unsqueeze(-1)).sum(1), fused.mean(1)], dim=-1)
        return self.classifier(feat)


# ===================== builders (from GAT_Mamba complexity cell) =====================
_nb = json.loads((_ROOT / 'GAT_Mamba.ipynb').read_text(encoding='utf-8'))
_src_cell = ''.join(_nb['cells'][2]['source'])
_i = _src_cell.find('def _src')
_j = _src_cell.find('REGISTRY = [')
_g = {
    'os': os, 'sys': sys, 're': re, 'ast': ast, 'json': json, 'time': time,
    'math': __import__('math'), 'np': np, 'pd': pd, 'torch': torch, 'nn': nn, 'F': F,
    'Dataset': Dataset, 'Data': Data, 'Batch': Batch,
    'GATv2Conv': GATv2Conv, 'GATConv': GATConv, 'SAGEConv': SAGEConv,
    'global_mean_pool': global_mean_pool, 'Path': Path, 'ROOT': ROOT,
    'WINDOW': WINDOW, 'NUM_CLASSES': NUM_CLASSES, 'SKIP_FUNCS': SKIP_FUNCS,
    'SECOND': SECOND, 'statistics': __import__('statistics'),
}
exec(_src_cell[_i:_j], _g)


def load_ns(nb_path, keep_ids=None):
    ns = {
        'torch': torch, 'nn': nn, 'F': F, 'np': np, 'Dataset': Dataset,
        'os': os, 'math': __import__('math'), 'Path': Path,
        'Data': Data, 'Batch': Batch,
        'GATv2Conv': GATv2Conv, 'GATConv': GATConv, 'SAGEConv': SAGEConv,
        'MessagePassing': None, 'global_mean_pool': global_mean_pool,
        'MAMBA_AVAILABLE': False, 'Mamba': None,
        'SSM_SCAN_CUDA': False, '_ssm_scan_fn': None,
        'GATAC_TEMP': 1.0, 'D_MODEL': 16, 'D_HIDDEN': 32, 'WINDOW_SIZE': 48,
        'CM_PATH': str(_ROOT / 'cm.png'), 'OUT_DIR': str(_ROOT),
        'RESULT_PATH': str(_ROOT / 'r.txt'), 'CKPT_PATH': str(_ROOT / 'x.pth'),
        'DATA_CSV': str(DATA_CSV), 'DEVICE': DEVICE, 'WINDOW': WINDOW,
        'NUM_CLASSES': NUM_CLASSES, 'pd': pd, 'warnings': warnings,
        'tqdm': (lambda x, **k: x),
        'plt': type('P', (), {'savefig': lambda *a, **k: None, 'close': lambda *a, **k: None,
                             'figure': lambda *a, **k: None, 'tight_layout': lambda *a, **k: None})(),
        'sns': type('S', (), {'heatmap': lambda *a, **k: None})(),
        'StandardScaler': StandardScaler, 'LabelEncoder': LabelEncoder,
        'accuracy_score': accuracy_score, 'f1_score': f1_score,
        'classification_report': (lambda *a, **k: ''),
        'confusion_matrix': (lambda *a, **k: np.zeros((1, 1))),
    }
    try:
        import pywt
        ns['pywt'] = pywt
        ns['PYWT_OK'] = True
    except Exception:
        ns['PYWT_OK'] = False
    try:
        from torch_geometric.nn import MessagePassing as MP
        ns['MessagePassing'] = MP
    except Exception:
        pass
    nb = json.loads(Path(nb_path).read_text(encoding='utf-8'))
    for c in nb.get('cells', []):
        if c.get('cell_type') != 'code':
            continue
        src = _g['_src'](c)
        cid = c.get('id', '')
        if keep_ids is not None and cid not in keep_ids:
            continue
        if ('轻量化对比' in src[:160] or '复杂度实测' in src[:160]
                or cid == 'cb0a65c4' or '提取 class1' in src or cid == '8fbb14cd'):
            continue
        try:
            chunk = _g['extract_defs'](src)
        except SyntaxError:
            continue
        if chunk.strip():
            try:
                exec(chunk, ns)
            except Exception:
                pass
    return ns


_g['load_ns'] = load_ns


def make_ultralite(device_dims, n_cls, soft):
    return InnovativeGATWithMambaUltraLite(
        device_dims, 16, 32, n_cls, gat_heads=1, dropout=0.1, use_soft_share=soft)


def make_baseline_models(device_dims, n_cls, classes):
    """Instantiate fresh models (random init) matching paper sizes from ckpts."""
    models = {}
    # CKAN
    ck = torch.load(_ROOT / 'best_ckan_bilstm.pth', map_location='cpu', weights_only=False)
    ns = load_ns(_ROOT / 'ckan-biLSTM.ipynb')
    pc = ck.get('paper_cfg', {})
    models['CKAN-BiLSTM'] = ('tensor', ns['CKANBiLSTM'](
        in_features=int(sum(device_dims)), num_classes=n_cls,
        conv1_out=int(pc.get('conv1_out', 32)), conv2_out=int(pc.get('conv2_out', 64)),
        kernel_size=int(pc.get('kernel_size', 3)),
        bilstm_hidden=int(pc.get('bilstm_hidden', 64)),
        fc_hidden=int(pc.get('fc_hidden', 64)), dropout=float(pc.get('dropout', 0.3)),
        grid_size=int(pc.get('grid_size', 5)), spline_s=int(pc.get('spline_s', 8))))
    # Wave
    ck = torch.load(_ROOT / 'best_wavemamba.pth', map_location='cpu', weights_only=False)
    ns = load_ns(_ROOT / 'wavemamba.ipynb')
    pc = ck.get('paper_cfg', {})
    models['WaveMamba'] = ('tensor', ns['WaveMamba'](
        in_dim=int(sum(device_dims)), num_classes=n_cls,
        d_model=int(pc.get('d_model', ck.get('d_model', 128))),
        d_state=int(pc.get('d_state', 16)),
        n_layers=int(pc.get('n_layers', 2)),
        dropout=float(pc.get('dropout', 0.3))))
    # TF
    ck = torch.load(_ROOT / 'best_transform_ids.pth', map_location='cpu', weights_only=False)
    ns = load_ns(_ROOT / 'transform_ids.ipynb')
    models['Transformer-IDS'] = ('tensor', ns['TransformerIDS'](
        in_dim=int(sum(device_dims)), num_classes=n_cls,
        d_model=int(ck.get('d_model', 256)), n_heads=int(ck.get('n_heads', 1)),
        n_layers=int(ck.get('n_layers', 1)), ffn_dim=int(ck.get('ffn_dim', 2048)),
        dropout=float(ck.get('dropout', 0.1))))
    # FeCo
    ck = torch.load(_ROOT / 'best_feco.pth', map_location='cpu', weights_only=False)
    ns = load_ns(_ROOT / 'Feco.ipynb')
    enc = ns['FeCoEncoder'](int(ck['in_dim']), 128, 256, latent_dim=int(ck['latent_dim']))
    head = ns['DownstreamHead'](enc, n_cls)
    models['FeCo'] = ('feco', head, np.asarray(ck['feat_idx'], dtype=np.int64))
    # GRID
    ck = torch.load(_ROOT / 'best_grid.pth', map_location='cpu', weights_only=False)
    ns = load_ns(_ROOT / 'GRID.ipynb')
    models['GRID'] = ('grid', ns['GRIDModel'](
        in_dim=int(ck.get('in_dim', 18)), num_classes=n_cls,
        d_model=int(ck.get('d_model', 64)), heads=int(ck.get('heads', 4)),
        dropout=float(ck.get('dropout', 0.2)), n_cascade=int(ck.get('n_cascade', 2))),
        ns['htgc_build_graph'], int(ck.get('n_seg', 6)), list(ck.get('device_dims', device_dims)))
    # MPGNN
    ck = torch.load(_ROOT / 'best_mpgnn.pth', map_location='cpu', weights_only=False)
    ns = load_ns(_ROOT / 'MPGNN.ipynb')
    models['MPGNN'] = ('mpgnn', ns['MPGNN'](
        in_dim=int(ck.get('in_dim', 18)), edge_dim=int(ck.get('edge_dim', 36)),
        num_classes=n_cls, d_model=int(ck.get('d_model', 128)),
        n_layers=int(ck.get('n_layers', 2)),
        n_coal=int(ck.get('n_coalitions', ck.get('n_coal', 4))),
        dropout=float(ck.get('dropout', 0.4))),
        ns['build_flow_graph'], int(ck.get('n_seg', 6)), list(ck.get('device_dims', device_dims)))
    # TCG
    ck = torch.load(_ROOT / 'best_tcg_ids.pth', map_location='cpu', weights_only=False)
    ns = load_ns(_ROOT / 'TCG-IDS.ipynb')
    models['TCG-IDS'] = ('tcg', ns['TCGIDS'](
        in_dim=int(ck.get('in_dim', 9)), num_classes=n_cls,
        d_model=int(ck.get('d_model', 64)), heads=int(ck.get('heads', 4)),
        dropout=float(ck.get('dropout', 0.2)), n_factor=int(ck.get('n_factor', 4))),
        ns['build_snapshot_graphs'], int(ck.get('n_snap', 8)), list(ck.get('device_dims', device_dims)))
    return models


# ===================== train / eval =====================
def macro_metrics(y, pred, n_cls=None):
    if n_cls is None:
        n_cls = int(max(y.max(), pred.max()) + 1)
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
def eval_tensor(model, X, y, device, batch=64, n_cls=None):
    model.eval()
    preds = []
    for i in range(0, len(X), batch):
        logits = model(torch.from_numpy(X[i:i + batch]).to(device))
        preds.append(logits.argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    return macro_metrics(y, pred, n_cls)


@torch.no_grad()
def eval_feco(model, X, y, feat_idx, device, batch=256, n_cls=None):
    model.eval()
    Z = np.concatenate([X.mean(1), X.std(1)], 1).astype(np.float32)[:, feat_idx]
    preds = []
    for i in range(0, len(Z), batch):
        preds.append(model(torch.from_numpy(Z[i:i + batch]).to(device)).argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    return macro_metrics(y, pred, n_cls)


@torch.no_grad()
def eval_graph(model, X, y, build_g, dims, n_seg, device, batch=24, n_cls=None):
    model.eval()
    preds = []
    for i in range(0, len(X), batch):
        gs = [build_g(X[j], dims, n_seg=n_seg).to(device) for j in range(i, min(i + batch, len(X)))]
        preds.append(model(Batch.from_data_list(gs)).argmax(1).cpu().numpy())
    return macro_metrics(y, np.concatenate(preds), n_cls)


@torch.no_grad()
def eval_tcg(model, X, y, build_snaps, dims, n_snap, device, batch=12, n_cls=None):
    model.eval()
    preds = []
    for i in range(0, len(X), batch):
        snaps = [[] for _ in range(n_snap)]
        for j in range(i, min(i + batch, len(X))):
            for s, g in enumerate(build_snaps(X[j], dims, n_snap=n_snap)):
                snaps[s].append(g.to(device))
        pred = model([Batch.from_data_list(s) for s in snaps]).argmax(1).cpu().numpy()
        preds.append(pred)
    return macro_metrics(y, np.concatenate(preds), n_cls)


def train_tensor(model, Xtr, ytr, Xva, yva, n_cls, device, epochs=EPOCHS, patience=PATIENCE):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    cw = class_weight_from_y(ytr, n_cls).to(device)
    best_f1, best_state, bad = -1.0, None, 0
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=BATCH, shuffle=True, drop_last=False)
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb, weight=cw)
            loss.backward()
            opt.step()
        vf1 = eval_tensor(model, Xva, yva, device, n_cls=n_cls)['f1']
        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, best_f1


def train_feco(model, Xtr, ytr, Xva, yva, feat_idx, n_cls, device):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    cw = class_weight_from_y(ytr, n_cls).to(device)
    Ztr = np.concatenate([Xtr.mean(1), Xtr.std(1)], 1).astype(np.float32)[:, feat_idx]
    Zva = np.concatenate([Xva.mean(1), Xva.std(1)], 1).astype(np.float32)[:, feat_idx]
    best_f1, best_state, bad = -1.0, None, 0
    loader = DataLoader(TensorDataset(torch.from_numpy(Ztr), torch.from_numpy(ytr)),
                        batch_size=BATCH, shuffle=True)
    for ep in range(1, EPOCHS + 1):
        model.train()
        for zb, yb in loader:
            zb, yb = zb.to(device), yb.to(device)
            opt.zero_grad()
            F.cross_entropy(model(zb), yb, weight=cw).backward()
            opt.step()
        # val
        model.eval()
        with torch.no_grad():
            pred = []
            for i in range(0, len(Zva), 256):
                pred.append(model(torch.from_numpy(Zva[i:i + 256]).to(device)).argmax(1).cpu().numpy())
            vf1 = float(f1_score(yva, np.concatenate(pred), average='macro', zero_division=0))
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


def train_graph(model, Xtr, ytr, Xva, yva, build_g, dims, n_seg, n_cls, device, kind='grid'):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    cw = class_weight_from_y(ytr, n_cls).to(device)
    best_f1, best_state, bad = -1.0, None, 0
    # subsample train for graph speed
    rng = np.random.RandomState(SEED)
    n_keep = min(len(Xtr), 8000)
    idx = rng.choice(len(Xtr), n_keep, replace=False)
    Xtr, ytr = Xtr[idx], ytr[idx]
    n_va = min(len(Xva), 2000)
    idxv = rng.choice(len(Xva), n_va, replace=False)
    Xva_s, yva_s = Xva[idxv], yva[idxv]

    for ep in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(Xtr))
        for i in range(0, len(order), BATCH_G):
            bi = order[i:i + BATCH_G]
            gs = [build_g(Xtr[j], dims, n_seg=n_seg).to(device) for j in bi]
            yb = torch.from_numpy(ytr[bi]).to(device)
            opt.zero_grad()
            F.cross_entropy(model(Batch.from_data_list(gs)), yb, weight=cw).backward()
            opt.step()
        vf1 = eval_graph(model, Xva_s, yva_s, build_g, dims, n_seg, device, BATCH_G, n_cls=n_cls)['f1']
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


def train_tcg(model, Xtr, ytr, Xva, yva, build_snaps, dims, n_snap, n_cls, device):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    cw = class_weight_from_y(ytr, n_cls).to(device)
    best_f1, best_state, bad = -1.0, None, 0
    rng = np.random.RandomState(SEED)
    n_keep = min(len(Xtr), 6000)
    idx = rng.choice(len(Xtr), n_keep, replace=False)
    Xtr, ytr = Xtr[idx], ytr[idx]
    n_va = min(len(Xva), 1500)
    idxv = rng.choice(len(Xva), n_va, replace=False)
    Xva_s, yva_s = Xva[idxv], yva[idxv]
    bs = 12
    for ep in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(Xtr))
        for i in range(0, len(order), bs):
            bi = order[i:i + bs]
            snaps = [[] for _ in range(n_snap)]
            for j in bi:
                for s, g in enumerate(build_snaps(Xtr[j], dims, n_snap=n_snap)):
                    snaps[s].append(g.to(device))
            yb = torch.from_numpy(ytr[bi]).to(device)
            opt.zero_grad()
            F.cross_entropy(model([Batch.from_data_list(s) for s in snaps]), yb, weight=cw).backward()
            opt.step()
        vf1 = eval_tcg(model, Xva_s, yva_s, build_snaps, dims, n_snap, device, bs, n_cls=n_cls)['f1']
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


def run_fold_all_models(X_tr, y_tr, X_va, y_va, X_te, y_te, device_dims, n_cls, classes, fold_label=''):
    """Train/eval 8 models on masked train/val; test on full features."""
    fold_scores = {}
    if fold_label:
        print(f'\n######## {fold_label} ########')

    m = make_ultralite(device_dims, n_cls, True)
    m, _ = train_tensor(m, X_tr, y_tr, X_va, y_va, n_cls, DEVICE)
    fold_scores['Ours'] = eval_tensor(m, X_te, y_te, DEVICE, n_cls=n_cls)
    print(f'  Ours Target-F1={fold_scores["Ours"]["f1"]:.4f}')
    del m
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    bas = make_baseline_models(device_dims, n_cls, classes)
    for name, spec in bas.items():
        kind = spec[0]
        t0 = time.time()
        if kind == 'tensor':
            model = spec[1]
            model, _ = train_tensor(model, X_tr, y_tr, X_va, y_va, n_cls, DEVICE)
            sc = eval_tensor(model, X_te, y_te, DEVICE, n_cls=n_cls)
            del model
        elif kind == 'feco':
            model, feat_idx = spec[1], spec[2]
            model, _ = train_feco(model, X_tr, y_tr, X_va, y_va, feat_idx, n_cls, DEVICE)
            sc = eval_feco(model, X_te, y_te, feat_idx, DEVICE, n_cls=n_cls)
            del model
        elif kind in ('grid', 'mpgnn'):
            model, build_g, n_seg, dims = spec[1], spec[2], spec[3], spec[4]
            model, _ = train_graph(model, X_tr, y_tr, X_va, y_va, build_g, dims, n_seg, n_cls, DEVICE)
            sc = eval_graph(model, X_te, y_te, build_g, dims, n_seg, DEVICE, n_cls=n_cls)
            del model
        elif kind == 'tcg':
            model, build_snaps, n_snap, dims = spec[1], spec[2], spec[3], spec[4]
            model, _ = train_tcg(model, X_tr, y_tr, X_va, y_va, build_snaps, dims, n_snap, n_cls, DEVICE)
            sc = eval_tcg(model, X_te, y_te, build_snaps, dims, n_snap, DEVICE, n_cls=n_cls)
            del model
        else:
            raise ValueError(kind)
        fold_scores[name] = sc
        print(f'  {name} Target-F1={sc["f1"]:.4f} ({time.time()-t0:.0f}s)')
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    return fold_scores


# ===================== prepare data =====================
METRIC_KEYS = [('acc', 'Accuracy'), ('prec', 'Precision'), ('rec', 'Recall'), ('f1', 'F1')]


def _agg_fold(results, model, fold_keys, key):
    vals = np.array([_fold_metric_val(results[model][fk], key) for fk in fold_keys], dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=1))


def main_lopo():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ckpt0 = torch.load(CKPT_OURS, map_location='cpu', weights_only=False)
    classes = list(ckpt0['classes'])
    n_cls = len(classes)
    df = pd.read_csv(DATA_CSV, low_memory=False)
    df['label_encoded'] = df['final_type'].map({c: i for i, c in enumerate(classes)}).astype(np.int64)
    normal_idx = classes.index('Normal')
    df_train = df[df.split == 'train'].reset_index(drop=True)
    df_test = df[df.split == 'test'].reset_index(drop=True)
    train_ds = EdgeIIoTFusionDataset(
        df_train, WINDOW, TRAIN_STEP, fit_scaler=True,
        max_windows=MAX_TRAIN, min_purity=MIN_PURITY, normal_idx=normal_idx)
    test_ds = EdgeIIoTFusionDataset(
        df_test, WINDOW, EVAL_STEP, scaler=train_ds.scaler,
        max_windows=MAX_EVAL, min_purity=MIN_PURITY, normal_idx=normal_idx)
    X_all_tr, y_all_tr = train_ds.samples, train_ds.labels
    X_te, y_te = test_ds.samples, test_ds.labels
    device_dims = list(train_ds.device_dims)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    tr_idx, va_idx = next(sss.split(np.zeros(len(y_all_tr)), y_all_tr))
    X_tr0, y_tr = X_all_tr[tr_idx], y_all_tr[tr_idx]
    X_va0, y_va = X_all_tr[va_idx], y_all_tr[va_idx]
    print(f'device={DEVICE} train={len(y_tr)} val={len(y_va)} test={len(y_te)} dims={device_dims}')

    SOURCE = {}
    pred_map = {
        'CKAN-BiLSTM': 'ckan_test_pred.npz',
        'GRID': 'grid_test_pred.npz',
        'WaveMamba': 'wavemamba_test_pred.npz',
        'Transformer-IDS': 'transform_ids_test_pred.npz',
        'FeCo': 'feco_test_pred.npz',
        'MPGNN': 'mpgnn_test_pred.npz',
        'TCG-IDS': 'tcg_ids_test_pred.npz',
        'Ours': 'ultralite_test_pred.npz',
        'UltraLite-Baseline': 'ultralite_baseline_test_pred.npz',
    }
    for name, fn in pred_map.items():
        p = _TMP / fn
        if p.is_file():
            pack = np.load(p)
            SOURCE[name] = float(f1_score(pack['y_true'], pack['y_pred'], average='macro', zero_division=0))
        else:
            SOURCE[name] = float('nan')
    print('Source F1:', {k: round(v, 4) for k, v in SOURCE.items()})

    MODEL_NAMES = [
        'Ours',
        'CKAN-BiLSTM', 'WaveMamba', 'Transformer-IDS', 'FeCo',
        'GRID', 'MPGNN', 'TCG-IDS',
    ]

    results = {m: {} for m in MODEL_NAMES}

    for p, pname in enumerate(PROTOCOL_GROUPS):
        cache_p = CACHE_DIR / f'fold_{pname}.npz'
        if cache_p.is_file():
            pack = np.load(cache_p, allow_pickle=True)
            for m in MODEL_NAMES:
                met = _cache_load_fold(pack, m)
                if met is not None:
                    results[m][pname] = met
            print(f'[cache] fold={pname} loaded')
            continue

        X_tr = zero_protocol(X_tr0, device_dims, p)
        X_va = zero_protocol(X_va0, device_dims, p)
        fold_scores = run_fold_all_models(
            X_tr, y_tr, X_va, y_va, X_te, y_te, device_dims, n_cls, classes,
            fold_label=f'LOPO fold hold-out protocol = {pname}')
        np.savez(cache_p, **_cache_save_fold(fold_scores))
        for m, sc in fold_scores.items():
            results[m][pname] = sc

    rows = []
    metric_rows = []
    for m in MODEL_NAMES:
        f1_scores = np.array([_fold_metric_val(results[m][p], 'f1') for p in PROTOCOL_GROUPS], dtype=np.float64)
        src = SOURCE.get(m, float('nan'))
        rows.append({
            'model': m,
            'Source_F1': src,
            'Target_mean': float(f1_scores.mean()),
            'Target_std': float(f1_scores.std(ddof=1)),
            'Target_worst': float(f1_scores.min()),
            'Gap_mean': float(src - f1_scores.mean()) if np.isfinite(src) else float('nan'),
            'Gap_worst': float(src - f1_scores.min()) if np.isfinite(src) else float('nan'),
            **{f'T_{p}': _fold_metric_val(results[m][p], 'f1') for p in PROTOCOL_GROUPS},
        })
        mrow = {'model': m}
        has_full = all(isinstance(results[m][p], dict) and 'acc' in results[m][p] for p in PROTOCOL_GROUPS)
        for k, label in METRIC_KEYS:
            if has_full or k == 'f1':
                mu, sd = _agg_fold(results, m, PROTOCOL_GROUPS, k)
                mrow[f'{label}_mean'] = mu
                mrow[f'{label}_std'] = sd
        metric_rows.append(mrow)

    df_sum = pd.DataFrame(rows).set_index('model')
    df_sum['score'] = df_sum['Target_mean'] - 0.25 * df_sum['Gap_mean'].fillna(0)
    show = df_sum.sort_values('score', ascending=False)
    cols = ['Source_F1', 'Target_mean', 'Target_std', 'Target_worst', 'Gap_mean', 'Gap_worst']

    print('\n' + '=' * 88)
    print('LOPO: train w/o protocol-p features → test full features')
    print(show[cols].round(4).to_string())
    print('=' * 88)
    print('\nPer-protocol Target F1:')
    print(show[[f'T_{p}' for p in PROTOCOL_GROUPS]].round(4).to_string())

    ours = df_sum.loc['Ours']
    print('\nOurs - others (dTarget_mean / dWorst / dGap↓ flipped):')
    for m, row in df_sum.iterrows():
        if m == 'Ours':
            continue
        print(f'  vs {m:20s}  dMean={ours.Target_mean-row.Target_mean:+.4f}  '
              f'dWorst={ours.Target_worst-row.Target_worst:+.4f}  '
              f'dGap={row.Gap_mean-ours.Gap_mean:+.4f}')

    lines = [
        'Leave-One-Protocol-Out (LOPO) on X-IIoTID fusion',
        'Train: zero held-out protocol features | Test: full features (unseen channel)',
        f'epochs={EPOCHS} patience={PATIENCE} max_train={MAX_TRAIN} seed={SEED}',
        '',
        show[cols].to_string(float_format=lambda x: f'{x:.6f}'),
        '',
        'Per-protocol Target F1:',
        show[[f'T_{p}' for p in PROTOCOL_GROUPS]].to_string(float_format=lambda x: f'{x:.6f}'),
        '',
        'Delta Ours - other:',
    ]
    for m, row in df_sum.iterrows():
        if m == 'Ours':
            continue
        lines.append(
            f'  vs {m:20s}  dMean={ours.Target_mean-row.Target_mean:+.6f}  '
            f'dWorst={ours.Target_worst-row.Target_worst:+.6f}  '
            f'dGap(flip)={row.Gap_mean-ours.Gap_mean:+.6f}')
    OUT_TXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    show.to_csv(OUT_CSV)
    df_metrics = pd.DataFrame(metric_rows).set_index('model')
    df_metrics.sort_values('F1_mean', ascending=False).to_csv(OUT_METRICS_CSV)
    print(f'\nsaved {OUT_TXT}')
    print(f'saved {OUT_CSV}')
    print(f'saved {OUT_METRICS_CSV}')


if __name__ == '__main__':
    main_lopo()
