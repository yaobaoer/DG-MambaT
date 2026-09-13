# -*- coding: utf-8 -*-
"""
Cross-device fusion comparison (IoT weather fusion, 7 devices).

Ours:        activity-weighted consensus + learnable soft fusion (γ)
Uniform:     uniform consensus + learnable γ (w/o activity weighting)
Low-Fixed-γ: activity-weighted + fixed γ=0.05 (insufficient fusion strength)
Private-only: γ≡0 (w/o cross-device fusion)
"""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager as fm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torch_geometric.nn import GATv2Conv

_ROOT = Path(r'E:\apt\YES\1')
_DATA_ROOT = Path(r'E:\apt\YES\2')
os.chdir(_ROOT)
_TMP = _ROOT / '_tmp'
_TMP.mkdir(exist_ok=True)
for k in ('TEMP', 'TMP', 'TMPDIR'):
    os.environ[k] = str(_TMP)
tempfile.tempdir = str(_TMP)
PHOTO_DIR = _ROOT / 'photo'
PHOTO_DIR.mkdir(exist_ok=True)
TNR_PATH = r'C:\Windows\Fonts\times.ttf'

# YES/2 UltraLite utilities (definitions only — no main training block)
_ul2_path = _DATA_ROOT / '_run_ultralite_train.py'
with open(_ul2_path, encoding='utf-8') as f:
    _ul2 = f.read()
_ns2 = {}
_split_marker = 'df = pd.read_csv(DATA_CSV, low_memory=False)'
exec(_ul2.split(_split_marker)[0], _ns2)

IoTFusionDataset = _ns2['IoTFusionDataset']
SimpleMamba = _ns2['SimpleMamba']
EMA = _ns2['EMA']
evaluate = _ns2['evaluate']
train_lite_epoch = _ns2['train_lite_epoch']
collect_logits = _ns2['collect_logits']
stratified_timegroup_split = _ns2['stratified_timegroup_split']
LabelEncoder = _ns2['LabelEncoder']
fc = _ns2['fc']

DATA_CSV = _DATA_ROOT / 'iot_fusion_weather_all.csv'
OUT_DIR = _ROOT / 'fusion_method_device_ckpts'
OUT_DIR.mkdir(exist_ok=True)
CKPT_OURS = OUT_DIR / 'fusion_device_ours.pth'

WINDOW, TRAIN_STEP, EVAL_STEP = 128, 1, 2
BATCH, EPOCHS, LR, PATIENCE = 64, 50, 1e-3, 5
DROPOUT = 0.1
LAM_KD, LAM_F1, EMA_DEC, RAMP = 0.65, 0.15, 0.999, 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FORCE = os.environ.get('FORCE_FUSION_DEVICE_TRAIN', '0') == '1'
FORCE_OURS = os.environ.get('FORCE_FUSION_DEVICE_OURS', '0') == '1'

METRICS = ['Accuracy', 'Precision', 'Recall', 'F1']
LINE_MODELS = [
    ('Uniform', '#4C78A8', 'o', 1.55, 7.5),
    ('Low-Fixed-γ', '#59A14F', '^', 1.55, 7.5),
    ('Private-only', '#B07AA1', 's', 1.55, 6.5),
    ('Ours', '#E15759', '*', 2.35, 13.5),
]

DROP_DEVICE_SETTINGS = [
    ('Full', 0),
    ('Drop1-weakest', 1),
    ('Drop2-weakest', 2),
    ('Drop3-weakest', 3),
    ('Drop4-weakest', 4),
]

FUSION_VARIANTS = [
    ('Ours', 'ours', CKPT_OURS),
    ('Uniform', 'uniform', OUT_DIR / 'fusion_device_uniform.pth'),
    ('Low-Fixed-γ', 'low_fixed', OUT_DIR / 'fusion_device_low_fixed005.pth'),
    ('Private-only', 'gamma_zero', OUT_DIR / 'fusion_device_private.pth'),
]


def tnr_prop(size):
    if os.path.isfile(TNR_PATH):
        fm.fontManager.addfont(TNR_PATH)
        return fm.FontProperties(fname=TNR_PATH, size=size)
    return fm.FontProperties(family='Times New Roman', size=size)


def setup_ph_style():
    if os.path.isfile(TNR_PATH):
        fm.fontManager.addfont(TNR_PATH)
        tnr_name = fm.FontProperties(fname=TNR_PATH).get_name()
    else:
        tnr_name = 'Times New Roman'
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': [tnr_name, 'Times New Roman', 'Times'],
        'axes.unicode_minus': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    })


def plot_fusion_drop_stress_lines(df_out, drop_settings, xlabel, out_stem):
    setup_ph_style()
    drop_settings = [(s, d) for s, d in drop_settings if d <= 4]
    xs = np.array([d for _, d in drop_settings])
    panels = [
        ('Accuracy', 'Accuracy', '(a)'),
        ('Precision', 'Precision', '(b)'),
        ('Recall', 'Recall', '(c)'),
        ('F1', 'F1', '(d)'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))
    handles = []
    for ax, (mkey, ylabel, caption) in zip(axes.flatten(), panels):
        ax.grid(True, linestyle='-', linewidth=0.5, color='#D8D8D8', alpha=0.95)
        ax.set_axisbelow(True)
        all_vals = []
        for name, color, marker, lw, ms in LINE_MODELS:
            ys = []
            for setting, _ in drop_settings:
                row = df_out[(df_out.method == name) & (df_out.setting == setting)]
                ys.append(float(row.iloc[0][mkey]) if len(row) else np.nan)
            ys = np.array(ys, dtype=float)
            all_vals.append(ys)
            z = 6 if name == 'Ours' else 4
            h = ax.plot(xs, ys, color=color, linestyle='-', linewidth=lw, marker=marker,
                        markersize=ms, markerfacecolor=color, markeredgecolor='white',
                        markeredgewidth=0.55, zorder=z)[0]
            if ax is axes[0, 0]:
                handles.append(h)
        flat = np.concatenate([v[~np.isnan(v)] for v in all_vals if len(v)])
        ymin = max(0.0, float(flat.min()) - 0.04)
        ymax = min(1.006, float(flat.max()) + 0.012)
        if mkey in ('Accuracy', 'Recall') and xs.max() <= 3 and ymin > 0.88:
            ymin = max(0.88, ymin)
        ax.set_xlim(xs.min() - 0.15, xs.max() + 0.15)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(int(x)) for x in xs], fontproperties=tnr_prop(10.5))
        ax.set_xlabel(xlabel, fontproperties=tnr_prop(11))
        ax.set_ylabel(ylabel, fontproperties=tnr_prop(12))
        ax.tick_params(direction='in', top=True, right=True, labelsize=10.5)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontproperties(tnr_prop(10.5))
        ax.text(0.5, -0.18, caption, transform=ax.transAxes, ha='center', va='top',
                fontproperties=tnr_prop(11))
    leg = fig.legend(handles, [n for n, _, _, _, _ in LINE_MODELS],
                     loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=4,
                     frameon=True, fancybox=False, edgecolor='#B0B0B0', framealpha=0.98,
                     handlelength=2.2, columnspacing=1.1, handletextpad=0.45, borderpad=0.38,
                     markerscale=0.85, prop=tnr_prop(9.5))
    for t in leg.get_texts():
        t.set_fontproperties(tnr_prop(9.5))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.10, top=0.88, wspace=0.22, hspace=0.34)
    rows = []
    for name, _, _, _, _ in LINE_MODELS:
        for setting, xdrop in drop_settings:
            row = df_out[(df_out.method == name) & (df_out.setting == setting)]
            if row.empty:
                continue
            r = row.iloc[0]
            rows.append({'method': name, 'dropped': xdrop, 'setting': setting,
                         **{k: r[k] for k in METRICS}})
    pd.DataFrame(rows).to_csv(_ROOT / f'{out_stem}_aprf.csv', index=False, encoding='utf-8-sig')
    for stem in (_ROOT / out_stem, PHOTO_DIR / out_stem):
        for ext in ('pdf', 'svg', 'png'):
            kw = dict(bbox_inches='tight', pad_inches=0.04)
            if ext == 'png':
                kw['dpi'] = 320
            fig.savefig(stem.with_suffix(f'.{ext}'), format=ext, **kw)
    plt.close(fig)
    print(f'saved {out_stem}.{{png,pdf,svg}} (+ photo/)')


def build_prior_weight(num_devices: int) -> torch.Tensor:
    pw = torch.full((num_devices, num_devices), 0.1)
    for i, j in [(4, 5), (5, 4), (2, 3), (3, 2), (4, 6), (6, 4)]:
        if i < num_devices and j < num_devices:
            pw[i, j] = 1.0
    for i, j in [(0, 1), (1, 0), (0, 2), (2, 0)]:
        if i < num_devices and j < num_devices:
            pw[i, j] = 0.2
    return pw


class FusionUltraLite(nn.Module):
    def __init__(self, device_dims, fusion_mode='ours',
                 d_model=16, d_hidden=32, num_classes=8, gat_heads=1, dropout=0.1):
        super().__init__()
        self.num_devices = len(device_dims)
        self.device_dims = device_dims
        self.d_model = d_model
        self.fusion_mode = fusion_mode
        self.device_projectors = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, d_model), nn.LayerNorm(d_model), nn.GELU())
            for dim in device_dims
        ])
        if fusion_mode not in ('gamma_zero', 'low_fixed'):
            self.share_logit = nn.Parameter(torch.tensor(-2.0))
        self.act_temp = nn.Parameter(torch.tensor(1.0))
        self.register_buffer('prior_weight', build_prior_weight(self.num_devices))
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

    def calibrate_cross_device(self, device_feats):
        if self.fusion_mode == 'gamma_zero':
            return device_feats
        act = device_feats.norm(dim=-1, keepdim=True)
        if self.fusion_mode == 'uniform':
            w = torch.full_like(act, 1.0 / self.num_devices)
        elif self.fusion_mode == 'max_act':
            idx = act.argmax(dim=2, keepdim=True)
            w = torch.zeros_like(act).scatter(2, idx, 1.0)
        else:
            temp = self.act_temp.clamp(0.2, 5.0)
            w = torch.softmax(act / temp, dim=2)
        shared = (device_feats * w).sum(dim=2, keepdim=True)
        if self.fusion_mode == 'low_fixed':
            gamma = device_feats.new_tensor(0.05)
        else:
            gamma = torch.sigmoid(self.share_logit)
        return (1.0 - gamma) * device_feats + gamma * shared

    def compute_dynamic_edges(self, pooled):
        B, N, D = pooled.shape
        dev = pooled.device
        ei = torch.tensor(
            [[i, j] for i in range(N) for j in range(N) if i != j],
            dtype=torch.long, device=dev).t().contiguous()
        normed = F.normalize(pooled, p=2, dim=-1)
        sim = torch.matmul(normed, normed.transpose(1, 2))
        dyn = torch.stack([sim[:, i, j] for i, j in zip(ei[0], ei[1])], dim=-1)
        fi, fj = pooled[:, ei[0]], pooled[:, ei[1]]
        alpha = self.dynamic_prior_gate(torch.cat([fi, fj], -1)).squeeze(-1)
        prior = self.prior_weight[ei[0], ei[1]].view(1, -1)
        agree = ((fi * fj).sum(-1) / (fi.norm(dim=-1) * fj.norm(dim=-1) + 1e-6)).clamp(0, 1)
        ew = ((alpha * prior + (1 - alpha) * dyn) * (0.5 + 0.5 * agree)).reshape(-1, 1)
        off = torch.arange(B, device=dev) * N
        return (ei.unsqueeze(1) + off.view(1, -1, 1)).reshape(2, -1), ew

    def forward(self, x):
        B, T, _ = x.shape
        feats, s = [], 0
        for proj, dim in zip(self.device_projectors, self.device_dims):
            feats.append(proj(x[:, :, s:s + dim]))
            s += dim
        device_feats = torch.stack(feats, dim=2)
        device_feats = self.calibrate_cross_device(device_feats)
        pooled = device_feats.mean(1)
        edge_index, edge_weights = self.compute_dynamic_edges(pooled)
        h = pooled.reshape(-1, self.d_model)
        h = self.n1(h + self.gat1(h, edge_index, edge_attr=edge_weights))
        out_seq = device_feats + h.reshape(B, 1, self.num_devices, self.d_model)
        fused = self.mamba(out_seq.reshape(B, T, -1))
        w = torch.softmax(self.attn(fused).squeeze(-1), dim=1)
        feat = torch.cat([(fused * w.unsqueeze(-1)).sum(1), fused.mean(1)], dim=-1)
        return self.classifier(feat)


def ramp(ep):
    t = min(1.0, ep / float(RAMP))
    return LAM_KD * math.exp(-5.0 * (1.0 - t) ** 2)


def drop_weakest(X, device_dims, n_drop=2):
    energies = []
    for p, dim in enumerate(device_dims):
        s = int(sum(device_dims[:p]))
        energies.append(np.linalg.norm(X[:, :, s:s + dim], axis=(1, 2)))
    order = np.argsort(np.stack(energies, 1), axis=1)
    X2 = X.copy()
    for i in range(len(X2)):
        for p in order[i, :n_drop]:
            s = int(sum(device_dims[:p]))
            X2[i, :, s:s + device_dims[p]] = 0.0
    return X2


def train_variant(model, train_loader, val_loader, le, n_cls, cw, cp, ckpt_path, tag):
    print(f'\n{"=" * 70}\nTrain fusion (device): {tag}\n{"=" * 70}')
    ema = EMA(model, decay=EMA_DEC)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=2e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2)
    best, bad = -1.0, 0
    labels = list(range(n_cls))
    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_f1 = train_lite_epoch(
            model, train_loader, opt, DEVICE, n_cls, cw, ema, lam_kd=ramp(ep), lam_f1=LAM_F1)
        bak = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)
        val = evaluate(model, val_loader, DEVICE, le, desc=f'Val-{tag}')
        model.load_state_dict(bak)
        print(f'Ep{ep:02d} loss={tr_loss:.4f} trF1={tr_f1:.4f} valF1={val["macro_f1"]:.4f}')
        if val['macro_f1'] > best + 1e-4:
            best, bad = val['macro_f1'], 0
            torch.save({
                'model_state': {k: v.cpu().clone() for k, v in ema.shadow.items()},
                'epoch': ep, 'val_macro_f1': best, 'classes': list(le.classes_),
                'device_dims': list(model.device_dims), 'fusion_mode': model.fusion_mode,
                'tag': tag,
            }, ckpt_path)
        else:
            bad += 1
            if bad >= PATIENCE:
                print('early stop')
                break
        sch.step(val['macro_f1'])
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    vl, vy = collect_logits(model, val_loader, DEVICE)
    adj_tau, pair_rules, _, _, _ = fc.tune_fpr_calibration(
        vl, vy, cp, labels, class_names=list(le.classes_))
    ckpt.update({'adj_tau': float(adj_tau), 'logit_bias': None, 'pair_rules': pair_rules})
    torch.save(ckpt, ckpt_path)
    return model, ckpt


def load_fusion_model(ckpt_path, fusion_mode, n_cls):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = FusionUltraLite(
        ckpt['device_dims'], fusion_mode=fusion_mode,
        num_classes=n_cls, dropout=DROPOUT,
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state'], strict=True)
    return model, ckpt


@torch.no_grad()
def eval_on_X(model, X, y, le, n_cls, cp, ckpt, tag):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False)
    log_prior = None
    if float(ckpt.get('adj_tau', 0)) != 0:
        log_prior = torch.log(torch.clamp(cp, min=1e-6)).to(DEVICE).view(1, -1)
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        logits = model(xb).float()
        pred = fc.apply_fpr_decision(
            logits, adj_tau=float(ckpt.get('adj_tau', 0)),
            log_prior=log_prior, logit_bias=ckpt.get('logit_bias'),
            pair_rules=ckpt.get('pair_rules'))
        preds.extend(pred.cpu().numpy())
        trues.extend(yb.numpy())
    yt, yp = np.asarray(trues), np.asarray(preds)
    return {
        'Accuracy': float((yt == yp).mean()),
        'Precision': float(precision_score(yt, yp, average='macro', zero_division=0)),
        'Recall': float(recall_score(yt, yp, average='macro', zero_division=0)),
        'F1': float(f1_score(yt, yp, average='macro', zero_division=0)),
    }


def main():
    print(f'device={DEVICE}  data={DATA_CSV}')
    df = pd.read_csv(DATA_CSV, low_memory=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    valid = df['final_type'].value_counts()[df['final_type'].value_counts() > 10].index
    df = df[df['final_type'].isin(valid)].copy()
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['final_type'])
    n_cls = len(le.classes_)
    df_tr, df_va, df_te = stratified_timegroup_split(df)

    tr = IoTFusionDataset(df_tr, WINDOW, TRAIN_STEP, fit_scaler=True, label_encoder=le)
    va = IoTFusionDataset(df_va, WINDOW, EVAL_STEP, scaler=tr.scaler, label_encoder=le)
    te = IoTFusionDataset(df_te, WINDOW, EVAL_STEP, scaler=tr.scaler, label_encoder=le)
    counts = np.maximum(np.bincount(tr.labels, minlength=n_cls).astype(float), 1.0)
    sw = 1.0 / np.power(counts[tr.labels], 0.35); sw /= sw.mean()
    train_loader = DataLoader(
        tr, batch_size=BATCH,
        sampler=torch.utils.data.WeightedRandomSampler(torch.DoubleTensor(sw), len(tr), True),
        num_workers=0, pin_memory=True)
    val_loader = DataLoader(va, batch_size=BATCH, shuffle=False)
    cw = torch.tensor(np.clip(np.sqrt(counts.sum() / (n_cls * counts)), 0.5, 2.0),
                      dtype=torch.float32, device=DEVICE)
    cp = torch.tensor(counts / counts.sum(), dtype=torch.float32)
    X_full, y = te.samples, te.labels
    dims = list(tr.device_dims)

    for name, mode, ckpt_path in FUSION_VARIANTS:
        if name == 'Ours':
            if ckpt_path.is_file() and not (FORCE_OURS or FORCE):
                print('skip train Ours (exists)')
                continue
        elif ckpt_path.is_file() and not FORCE:
            print(f'skip train {name} (exists)')
            continue
        m = FusionUltraLite(dims, fusion_mode=mode, num_classes=n_cls).to(DEVICE)
        train_variant(m, train_loader, val_loader, le, n_cls, cw, cp, ckpt_path, name)

    rows = []
    for name, mode, ckpt_path in FUSION_VARIANTS:
        if not ckpt_path.is_file():
            print(f'missing {ckpt_path}')
            continue
        if name == 'Ours':
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            m = FusionUltraLite(dims, fusion_mode='ours', num_classes=n_cls).to(DEVICE)
            m.load_state_dict(ckpt['model_state'], strict=True)
        else:
            m, ckpt = load_fusion_model(ckpt_path, mode, n_cls)
        for setting, kdrop in DROP_DEVICE_SETTINGS:
            Xv = X_full if kdrop == 0 else drop_weakest(X_full, dims, kdrop)
            met = eval_on_X(m, Xv, y, le, n_cls, cp, ckpt, f'{name}-{setting}')
            rows.append({'method': name, 'setting': setting, **met})
            print(f'{name:14s} {setting:14s}  Acc={met["Accuracy"]:.4f} P={met["Precision"]:.4f} '
                  f'R={met["Recall"]:.4f} F1={met["F1"]:.4f}')

    df_out = pd.DataFrame(rows)
    df_out.to_csv(_ROOT / 'fusion_method_device_compare.csv', index=False, encoding='utf-8-sig')
    plot_fusion_drop_stress_lines(
        df_out, DROP_DEVICE_SETTINGS, xlabel='Dropped devices', out_stem='fusion_drop_device')
    print('saved fusion_method_device_compare.csv')


if __name__ == '__main__':
    main()
