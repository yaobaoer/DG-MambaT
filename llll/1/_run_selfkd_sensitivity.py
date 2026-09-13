# ===== Mean-Teacher 自蒸馏超参敏感性分析 =====
# OAT 扫描: lam_kd / ema_decay / ramp_epochs / kd_temp
# 骨干: UltraLite；数据=YES/2 iot_fusion_weather_all.csv（7 设备）
# 配方对齐 YES/2 UltraLite：window=128 / train_step=1 / eval_step=2
# 不覆盖 best_gat_mamba_ultralite.pth
# 输出: selfkd_sensitivity_*.{csv,txt} + photo/selfkd_sensitivity_*.{png,pdf}

import os
import math
import time
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from torch_geometric.nn import GATv2Conv

_OUT_ROOT = os.path.abspath(os.getcwd())
for _cand in (_OUT_ROOT, '/root/1/YES/1', r'E:\apt\YES\1'):
    if os.path.isfile(os.path.join(_cand, '_run_selfkd_sensitivity.py')):
        _OUT_ROOT = _cand
        break
os.chdir(_OUT_ROOT)

_DATA_ROOT = None
for _cand in (
    os.path.join(_OUT_ROOT, '..', '2'),
    '/root/1/YES/2',
    r'E:\apt\YES\2',
):
    _p = os.path.abspath(_cand)
    if os.path.isfile(os.path.join(_p, 'iot_fusion_weather_all.csv')):
        _DATA_ROOT = _p
        break
assert _DATA_ROOT, '找不到 YES/2/iot_fusion_weather_all.csv'
_ROOT = _OUT_ROOT
_TMP = os.path.join(_OUT_ROOT, '_tmp')
os.makedirs(_TMP, exist_ok=True)
for _k in ('TEMP', 'TMP', 'TMPDIR'):
    os.environ[_k] = _TMP
tempfile.tempdir = _TMP

PHOTO = os.path.join(_OUT_ROOT, 'photo')
os.makedirs(PHOTO, exist_ok=True)
DATA_CSV = os.path.join(_DATA_ROOT, 'iot_fusion_weather_all.csv')
TNR_PATH = r'C:\Windows\Fonts\times.ttf'

DEVICE_PREFIXES = ['weather', 'modbus', 'thermostat', 'fridge',
                   'garage_door', 'gps', 'motion_light']

# ---- 对齐 YES/2 UltraLite 主实验数据协议 ----
WINDOW_SIZE, TRAIN_STEP, EVAL_STEP = 128, 1, 2
BATCH_SIZE, LR, DROPOUT = 64, 1e-3, 0.1
EPOCHS, PATIENCE = 50, 5
LAM_F1 = 0.15
D_MODEL, D_HIDDEN, GAT_HEADS = 16, 32, 1

DEFAULTS = dict(lam_kd=0.65, ema_decay=0.999, ramp_epochs=10, kd_temp=2.0)

GRID = {
    'lam_kd': [0.35, 0.50, 0.65, 0.80],
    'ema_decay': [0.990, 0.995, 0.999, 0.9995],
    'ramp_epochs': [0, 5, 10, 20],
    'kd_temp': [1.0, 2.0, 4.0],
}

_AXES_ONLY = [a.strip() for a in os.environ.get('SELFKD_AXES', '').split(',') if a.strip()]
if _AXES_ONLY:
    GRID = {k: v for k, v in GRID.items() if k in _AXES_ONLY}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tnr(size):
    if os.path.isfile(TNR_PATH):
        fm.fontManager.addfont(TNR_PATH)
        return fm.FontProperties(fname=TNR_PATH, size=size)
    return fm.FontProperties(family='DejaVu Serif', size=size)


class IoTFusionDataset(Dataset):
 
    def __init__(self, df, window_size=128, step=1, scaler=None, fit_scaler=False):
        self.window_size = window_size
        self.step = step
        self.devices = list(DEVICE_PREFIXES)
        self.device_dims = []
        self.all_feat_cols = []
        for dev in DEVICE_PREFIXES:
            cols = [c for c in df.columns if c.startswith(dev + '_') and
                    not c.endswith('_label') and not c.endswith('_type')]
            self.device_dims.append(len(cols))
            self.all_feat_cols.extend(cols)

        X = df[self.all_feat_cols].values.astype(np.float32)
        y_enc = df['label_encoded'].values.astype(np.int64)
        if scaler is None and fit_scaler:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        elif scaler is not None:
            self.scaler = scaler
            X = self.scaler.transform(X)
        else:
            self.scaler = None
        X = np.nan_to_num(np.clip(X, -8, 8), nan=0.0).astype(np.float32)

        samples, labels = [], []
        for i in range(0, len(X) - window_size + 1, step):
            samples.append(X[i:i + window_size])
            labels.append(int(y_enc[i + window_size - 1]))
        if not samples:
            raise ValueError('no windows')
        self.samples = np.stack(samples, axis=0)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.n_raw = len(self.labels)
        print(f'windows={len(self)} dims={self.device_dims} step={step} '
              f'classes={sorted(np.unique(self.labels).tolist())}')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.samples[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


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


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


class InnovativeGATWithMambaUltraLite(nn.Module):
    def __init__(self, device_dims, d_model=16, d_hidden=32, num_classes=8,
                 window_size=128, gat_heads=1, dropout=0.1):
        super().__init__()
        self.num_devices = len(device_dims)
        self.device_dims = device_dims
        self.d_model = d_model
        self.device_projectors = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, d_model), nn.LayerNorm(d_model), nn.GELU())
            for dim in device_dims
        ])
        self.share_logit = nn.Parameter(torch.tensor(-2.0))
        self.act_temp = nn.Parameter(torch.tensor(1.0))
        prior_weight = torch.zeros((self.num_devices, self.num_devices))
        for i, j in [(4, 5), (5, 4), (2, 3), (3, 2), (4, 6), (6, 4)]:
            if i < self.num_devices and j < self.num_devices:
                prior_weight[i, j] = 1.0
        for i, j in [(0, 1), (1, 0), (0, 2), (2, 0)]:
            if i < self.num_devices and j < self.num_devices:
                prior_weight[i, j] = 0.2
        prior_weight[prior_weight == 0] = 0.1
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
            nn.LayerNorm(mamba_dim * 2),
            nn.Linear(mamba_dim * 2, num_classes))

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
        agree = ((fi * fj).sum(-1) / (fi.norm(dim=-1) * fj.norm(dim=-1) + 1e-6)).clamp(0, 1)
        ew = ((alpha * prior + (1 - alpha) * dyn) * (0.5 + 0.5 * agree)).reshape(-1, 1)
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


def focal_loss(logits, labels, alpha=0.25, gamma=2.0, weight=None):
    ce = F.cross_entropy(logits.float(), labels, reduction='none',
                         weight=weight, label_smoothing=0.015)
    pt = torch.exp(-ce.detach())
    return (alpha * (1 - pt).clamp(0, 1) ** gamma * ce).mean()


def soft_f1_loss(logits, y, n_cls, eps=1e-6):
    p = F.softmax(logits.float(), 1)
    t = F.one_hot(y, n_cls).float()
    tp = (p * t).sum(0)
    fp = (p * (1 - t)).sum(0)
    fn = ((1 - p) * t).sum(0)
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return 1 - f1.mean()


def kd_loss(student_logits, teacher_logits, T=2.0):
    s = F.log_softmax(student_logits.float() / T, dim=1)
    t = F.softmax(teacher_logits.float() / T, dim=1)
    return F.kl_div(s, t, reduction='batchmean') * (T * T)


def stratified_timegroup_split(df):
    
    df = df.copy()
    df['time_group'] = df.index // 2000
    train_indices, val_indices, test_indices = [], [], []
    for _, group in df.groupby('time_group'):
        if len(group) < 3:
            train_indices.extend(group.index.tolist())
            continue
        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            tr_val_idx, test_idx = next(sss.split(group, group['label_encoded']))
        except ValueError:
            tr_val_idx = np.arange(len(group))
            test_idx = np.array([], dtype=int)
        if len(tr_val_idx) > 1:
            try:
                sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                sub = group.iloc[tr_val_idx]
                train_idx, val_idx = next(sss2.split(sub, sub['label_encoded']))
                train_indices.extend(group.index[tr_val_idx][train_idx].tolist())
                val_indices.extend(group.index[tr_val_idx][val_idx].tolist())
            except ValueError:
                train_indices.extend(group.index[tr_val_idx].tolist())
        else:
            train_indices.extend(group.index[tr_val_idx].tolist())
        if len(test_idx):
            test_indices.extend(group.index[test_idx].tolist())
    return (
        df.loc[train_indices].sort_index().reset_index(drop=True),
        df.loc[val_indices].sort_index().reset_index(drop=True),
        df.loc[test_indices].sort_index().reset_index(drop=True),
    )


def consistency_rampup(epoch, ramp_epochs, max_w):
    
    t = 1.0 if ramp_epochs <= 0 else min(1.0, float(epoch) / float(ramp_epochs))
    return float(max_w * math.exp(-5.0 * (1.0 - t) ** 2))


@torch.no_grad()
def eval_macro(model, loader):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x)
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(y.numpy())
    yt, yp = np.asarray(trues), np.asarray(preds)
    return dict(
        acc=float(accuracy_score(yt, yp)),
        precision=float(precision_score(yt, yp, average='macro', zero_division=0)),
        recall=float(recall_score(yt, yp, average='macro', zero_division=0)),
        f1=float(f1_score(yt, yp, average='macro', zero_division=0)),
    )


def train_one(cfg, train_loader, val_loader, test_loader, n_cls, class_weight, device_dims):
   
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    model = InnovativeGATWithMambaUltraLite(
        device_dims=device_dims, d_model=D_MODEL, d_hidden=D_HIDDEN,
        num_classes=n_cls, window_size=WINDOW_SIZE, gat_heads=GAT_HEADS,
        dropout=DROPOUT).to(DEVICE)
    use_kd = float(cfg['lam_kd']) > 0
    ema = EMA(model, decay=float(cfg['ema_decay'])) if use_kd else None
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=2, min_lr=1e-6)

    best_f1, best_state, bad, best_ep = -1.0, None, 0, 0
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        lam_now = consistency_rampup(ep, int(cfg['ramp_epochs']), float(cfg['lam_kd'])) if use_kd else 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            if ema is not None and lam_now > 0:
                with torch.no_grad():
                    bak = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    ema.copy_to(model)
                    model.eval()
                    t_logits = model(x)
                    model.train()
                    model.load_state_dict(bak, strict=True)
            s_logits = model(x)
            hard = focal_loss(s_logits, y, weight=class_weight) + LAM_F1 * soft_f1_loss(s_logits, y, n_cls)
            if ema is not None and lam_now > 0:
                loss = (1.0 - lam_now) * hard + lam_now * kd_loss(
                    s_logits, t_logits, T=float(cfg['kd_temp']))
            else:
                loss = hard
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema is not None:
                ema.update(model)

        if ema is not None:
            bak = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
            val = eval_macro(model, val_loader)
            model.load_state_dict(bak, strict=True)
            state = {k: v.detach().cpu().clone() for k, v in ema.shadow.items()}
        else:
            val = eval_macro(model, val_loader)
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"  ep={ep}/{EPOCHS} lam={lam_now:.3f} val_f1={val['f1']:.4f} lr={opt.param_groups[0]['lr']:.2e}")
        if val['f1'] > best_f1 + 1e-4:
            best_f1, best_state, bad, best_ep = val['f1'], state, 0, ep
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f'  early stop @ ep={ep}')
                break
        sch.step(val['f1'])

    model.load_state_dict(best_state, strict=True)
    val = eval_macro(model, val_loader)
    test = eval_macro(model, test_loader)
    secs = time.time() - t0
    return dict(
        best_epoch=best_ep, seconds=round(secs, 1),
        val_acc=val['acc'], val_p=val['precision'], val_r=val['recall'], val_f1=val['f1'],
        test_acc=test['acc'], test_p=test['precision'], test_r=test['recall'], test_f1=test['f1'],
    )


def hp_key(cfg):
    return (float(cfg['lam_kd']), float(cfg['ema_decay']),
            int(cfg['ramp_epochs']), float(cfg['kd_temp']))


def hp_id(cfg):
    return (f"lam={cfg['lam_kd']}|ema={cfg['ema_decay']}|"
            f"ramp={cfg['ramp_epochs']}|T={cfg['kd_temp']}")


def build_oat_cfgs():
    cfgs = []
    for axis, values in GRID.items():
        for v in values:
            cfg = dict(DEFAULTS)
            cfg[axis] = v
            cfg['axis'] = axis
            cfg['axis_value'] = v
            cfg['run_id'] = f'{axis}={v}'
            cfgs.append(cfg)
    return cfgs


def plot_axis(df_axis, axis, ylabel='Test Macro-F1'):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    xs = df_axis['axis_value'].values
    y_te = df_axis['test_f1'].values
    y_va = df_axis['val_f1'].values
    ax.plot(xs, y_va, 'o--', color='#4C78A8', label='Val F1', markersize=7)
    ax.plot(xs, y_te, 's-', color='#E15759', label='Test F1', markersize=7)
    def_v = DEFAULTS[axis]
    ax.axvline(def_v, color='#B0B0B0', ls=':', lw=1.2, label=f'default={def_v}')
    ax.set_xlabel(axis, fontproperties=tnr(11))
    ax.set_ylabel(ylabel, fontproperties=tnr(11))
    ax.set_title(f'Self-KD sensitivity: {axis}', fontproperties=tnr(12))
    ax.grid(True, ls='-', color='#E0E0E0', alpha=0.9)
    ax.legend(prop=tnr(9))
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(tnr(9.5))
    fig.tight_layout()
    stem = os.path.join(PHOTO, f'selfkd_sensitivity_{axis}')
    for ext in ('png', 'pdf'):
        fig.savefig(f'{stem}.{ext}', dpi=280 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)
    print('saved', stem + '.{png,pdf}')


def main():
    print('=' * 64)
    print('Self-KD sensitivity (OAT) on UltraLite / IoT-Weather')
    print(f'window={WINDOW_SIZE} step={TRAIN_STEP}/{EVAL_STEP}  '
          f'epochs={EPOCHS} patience={PATIENCE}')
    print('=' * 64)
    print('OUT', _OUT_ROOT)
    print('DATA', DATA_CSV)
    print('device', DEVICE)
    print('defaults', DEFAULTS)
    print('grid', {k: GRID[k] for k in GRID})

    df = pd.read_csv(DATA_CSV, low_memory=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    valid_types = df['final_type'].value_counts()[df['final_type'].value_counts() > 10].index
    df = df[df['final_type'].isin(valid_types)].copy()
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['final_type'].astype(str))
    classes = list(le.classes_)
    n_cls = len(classes)
    print('classes', n_cls, classes)

    df_train, df_val, df_test = stratified_timegroup_split(df)
    print(f'rows train/val/test={len(df_train)}/{len(df_val)}/{len(df_test)}')

    train_ds = IoTFusionDataset(
        df_train, window_size=WINDOW_SIZE, step=TRAIN_STEP, fit_scaler=True)
    val_ds = IoTFusionDataset(
        df_val, window_size=WINDOW_SIZE, step=EVAL_STEP, scaler=train_ds.scaler)
    test_ds = IoTFusionDataset(
        df_test, window_size=WINDOW_SIZE, step=EVAL_STEP, scaler=train_ds.scaler)

    counts = np.maximum(np.bincount(train_ds.labels, minlength=n_cls).astype(np.float64), 1.0)
    sw = 1.0 / np.sqrt(counts[train_ds.labels])
    sw = sw / sw.mean()
    sampler = torch.utils.data.WeightedRandomSampler(torch.DoubleTensor(sw), len(train_ds), True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)
    cw = np.clip(np.sqrt(counts.sum() / (n_cls * counts)), 0.5, 3.0)
    class_weight = torch.tensor(cw, dtype=torch.float32, device=DEVICE)

    cfgs = build_oat_cfgs()
    unique_keys = []
    seen_hp = set()
    for c in cfgs:
        k = hp_key(c)
        if k not in seen_hp:
            seen_hp.add(k)
            unique_keys.append(k)
    print(f'OAT axis-rows={len(cfgs)}  unique HP={len(unique_keys)}')

    cache_path = os.path.join(_ROOT, 'selfkd_sensitivity_cache.csv')
    done = {}
    if os.path.isfile(cache_path):
        cache = pd.read_csv(cache_path)
        for _, r in cache.iterrows():
            done[r['hp_id']] = r.to_dict()
        print(f'cache hit {len(done)} / {len(unique_keys)} unique HP')

    hp_metrics = {}
    for i, k in enumerate(unique_keys, 1):
        proto = dict(lam_kd=k[0], ema_decay=k[1], ramp_epochs=k[2], kd_temp=k[3])
        hid = hp_id(proto)
        print(f'\n[{i}/{len(unique_keys)}] {hid}')
        cached = done.get(hid)
        if cached is not None:
            tf1 = cached.get('test_f1')
            if not (isinstance(tf1, float) and np.isnan(tf1)):
                print('  skip (cache)')
                hp_metrics[k] = cached
                continue
        metrics = train_one(proto, train_loader, val_loader, test_loader,
                            n_cls, class_weight, train_ds.device_dims)
        rec = {**proto, 'hp_id': hid, **metrics}
        hp_metrics[k] = rec
        done[hid] = rec
        pd.DataFrame(list(done.values())).to_csv(cache_path, index=False)
        print(f"  >> val_f1={metrics['val_f1']:.4f} test_f1={metrics['test_f1']:.4f} "
              f"({metrics['seconds']}s)")

    rows = []
    for cfg in cfgs:
        rec = dict(hp_metrics[hp_key(cfg)])
        rec.update({k: cfg[k] for k in (
            'axis', 'axis_value', 'run_id', 'lam_kd', 'ema_decay', 'ramp_epochs', 'kd_temp')})
        rec['n_train'] = int(len(train_ds))
        rec['n_val'] = int(len(val_ds))
        rec['n_test'] = int(len(test_ds))
        rec['n_train_raw'] = int(train_ds.n_raw)
        rec['n_val_raw'] = int(val_ds.n_raw)
        rec['n_test_raw'] = int(test_ds.n_raw)
        rows.append(rec)

    df_all = pd.DataFrame(rows)
    summary_lines = [
        'selfkd_sensitivity_summary',
        f'defaults={DEFAULTS}',
        f'device=window={WINDOW_SIZE} step={TRAIN_STEP}/{EVAL_STEP} batch={BATCH_SIZE} lr={LR} '
        f'epochs={EPOCHS} patience={PATIENCE} data=iot_fusion_weather_all.csv',
        f'windows train/val/test={len(train_ds)}/{len(val_ds)}/{len(test_ds)} '
        f'raw={train_ds.n_raw}/{val_ds.n_raw}/{test_ds.n_raw}',
        'note=IoT-Weather 7-device; aligned with YES/2 UltraLite recipe',
        '',
    ]
    for axis in GRID:
        sub = df_all[df_all['axis'] == axis].copy()
        sub = sub.sort_values('axis_value')
        csv_p = os.path.join(_ROOT, f'selfkd_sensitivity_{axis}.csv')
        txt_p = os.path.join(_ROOT, f'selfkd_sensitivity_{axis}.txt')
        sub.to_csv(csv_p, index=False)
        with open(txt_p, 'w', encoding='utf-8') as f:
            f.write(f'axis={axis}\n')
            f.write(sub.to_string(index=False) + '\n')
        plot_axis(sub, axis)
        summary_lines.append(f'=== {axis} ===')
        summary_lines.append(sub[['axis_value', 'val_f1', 'test_f1', 'best_epoch']].to_string(index=False))
        summary_lines.append('')
        print(f'saved {csv_p}')

    all_csv = os.path.join(_ROOT, 'selfkd_sensitivity_all.csv')
    df_all.to_csv(all_csv, index=False)
    sum_p = os.path.join(_ROOT, 'selfkd_sensitivity_summary.txt')
    Path(sum_p).write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')
    print('saved', all_csv)
    print('saved', sum_p)
    print('\nDone. Formal UltraLite ckpt untouched.')
    return df_all


if __name__ == '__main__':
    main()
