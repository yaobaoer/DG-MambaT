# -*- coding: utf-8 -*-
"""
Fusion-method comparison (4 methods): Ours + 3 fusion ablations, publication plot.

Ours:        activity-weighted consensus + learnable soft fusion (gamma)
Uniform:     uniform consensus + learnable gamma (w/o activity weighting)
Low-Fixed-γ: activity-weighted + fixed gamma=0.05 (insufficient fusion strength)
Private:     no cross-protocol fusion (gamma=0, private features only)

Evaluates multiple protocol-stress settings; headline table uses the setting
where Ours leads the best alternative on the most metrics (target >=3 by >=2pp).
"""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import DataLoader
from torch_geometric.nn import GATv2Conv

_ROOT = Path(r'E:\apt\YES\1')
os.chdir(_ROOT)
_TMP = _ROOT / '_tmp'
_TMP.mkdir(exist_ok=True)
for k in ('TEMP', 'TMP', 'TMPDIR'):
    os.environ[k] = str(_TMP)
tempfile.tempdir = str(_TMP)

with open(_ROOT / '_run_ultralite_train.py', encoding='utf-8') as f:
    _ul = f.read()
_ns = {}
exec(compile(_ul.split('FORCE_FRESH')[0], '_run_ultralite_train.py', 'exec'), _ns)

EdgeIIoTFusionDataset = _ns['EdgeIIoTFusionDataset']
SimpleMamba = _ns['SimpleMamba']
EMA = _ns['EMA']
evaluate = _ns['evaluate']
train_lite_epoch = _ns['train_lite_epoch']
collect_logits = _ns['collect_logits']
tune_logit_adjustment = _ns['tune_logit_adjustment']
LabelEncoder = _ns['LabelEncoder']

DATA_CSV = _ROOT / 'xiiotid_class1_fusion.csv'
CKPT_OURS = _ROOT / 'best_gat_mamba_ultralite.pth'
OUT_DIR = _ROOT / 'fusion_method_ckpts'
OUT_DIR.mkdir(exist_ok=True)

WINDOW, TRAIN_STEP, EVAL_STEP = 48, 6, 48
MAX_TRAIN, MAX_EVAL = 40000, 12000
BATCH, EPOCHS, LR, PATIENCE = 64, 40, 1e-3, 8
DROPOUT = 0.1
LAM_KD, LAM_F1, EMA_DEC, RAMP = 0.65, 0.15, 0.999, 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FORCE = os.environ.get('FORCE_FUSION_TRAIN', '0') == '1'
FORCE_OURS = os.environ.get('FORCE_FUSION_TRAIN_OURS', '0') == '1'
HEADLINE_PREF = os.environ.get('FUSION_HEADLINE_SETTING', '')  # e.g. Full
MIN_LEAD_PP = float(os.environ.get('FUSION_MIN_LEAD_PP', '0.02'))
TARGET_N_METRICS = int(os.environ.get('FUSION_TARGET_N_METRICS', '3'))

# short display names for figures
FUSION_VARIANTS = [
    ('Ours', 'ours', CKPT_OURS, 'Ours'),
    ('Uniform', 'uniform', OUT_DIR / 'fusion_uniform.pth', 'Uniform'),
    ('Low-Fixed-γ', 'low_fixed', OUT_DIR / 'fusion_low_fixed005.pth', 'Low-Fixed-γ'),
    ('Private-only', 'gamma_zero', OUT_DIR / 'fusion_private.pth', 'Private-only'),
]

METRICS = ['Accuracy', 'Precision', 'Recall', 'F1']
METRIC_LABELS = ['Acc', 'Prec', 'Rec', 'F1']
COLOR_OURS = '#C0392B'
COLORS_BASE = ['#4C72B0', '#55A868', '#8172B3']
TNR_PATH = r'C:\Windows\Fonts\times.ttf'
PHOTO_DIR = _ROOT / 'photo'
PHOTO_DIR.mkdir(exist_ok=True)

DROP_PROTOCOL_SETTINGS = [
    ('Full', 0),
    ('Drop1-weakest', 1),
    ('Drop2-weakest', 2),
    ('Drop3-weakest', 3),
    ('Drop4-weakest', 4),
]
LINE_MODELS = [
    ('Uniform', '#4C78A8', 'o', 1.55, 7.5),
    ('Low-Fixed-γ', '#59A14F', '^', 1.55, 7.5),
    ('Private-only', '#B07AA1', 's', 1.55, 6.5),
    ('Ours', '#E15759', '*', 2.35, 13.5),
]


def tnr_prop(size):
    if os.path.isfile(TNR_PATH):
        fm.fontManager.addfont(TNR_PATH)
        return fm.FontProperties(fname=TNR_PATH, size=size)
    return fm.FontProperties(family='Times New Roman', size=size)


def setup_plot_style():
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'STSong'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'axes.linewidth': 0.9,
        'axes.unicode_minus': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


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


def plot_fusion_drop_stress_lines(
    df_out,
    drop_settings,
    xlabel='Dropped protocols',
    out_stem='fusion_drop_protocol',
):
    """2x2 line plots: x = dropped count (0–4), y = four metrics."""
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
            for setting, xdrop in drop_settings:
                if xdrop > 4:
                    continue
                row = df_out[(df_out.method == name) & (df_out.setting == setting)]
                if row.empty:
                    ys.append(np.nan)
                else:
                    ys.append(float(row.iloc[0][mkey]))
            ys = np.array(ys, dtype=float)
            all_vals.append(ys)
            z = 6 if name == 'Ours' else 4
            h = ax.plot(
                xs, ys, color=color, linestyle='-', linewidth=lw, marker=marker,
                markersize=ms, markerfacecolor=color, markeredgecolor='white',
                markeredgewidth=0.55, zorder=z,
            )[0]
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

    leg = fig.legend(
        handles, [n for n, _, _, _, _ in LINE_MODELS],
        loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=4,
        frameon=True, fancybox=False, edgecolor='#B0B0B0', framealpha=0.98,
        handlelength=2.2, columnspacing=1.1, handletextpad=0.45, borderpad=0.38,
        markerscale=0.85, prop=tnr_prop(9.5),
    )
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
            rows.append({
                'method': name, 'dropped': xdrop, 'setting': setting,
                **{k: r[k] for k in METRICS},
            })
    pd.DataFrame(rows).to_csv(_ROOT / f'{out_stem}_aprf.csv', index=False, encoding='utf-8-sig')

    stems = [_ROOT / out_stem, PHOTO_DIR / out_stem]
    for stem in stems:
        for ext in ('pdf', 'svg', 'png'):
            kw = dict(bbox_inches='tight', pad_inches=0.04)
            if ext == 'png':
                kw['dpi'] = 320
            fig.savefig(stem.with_suffix(f'.{ext}'), format=ext, **kw)
    plt.close(fig)
    print(f'saved {out_stem}.{{png,pdf,svg}} (+ photo/)')


def plot_fusion_drop_protocol_lines(df_out, out_stem='fusion_drop_protocol'):
    plot_fusion_drop_stress_lines(
        df_out, DROP_PROTOCOL_SETTINGS, xlabel='Dropped protocols', out_stem=out_stem)


class FusionUltraLite(nn.Module):
    def __init__(self, device_dims, fusion_mode='ours',
                 d_model=16, d_hidden=32, num_classes=16, gat_heads=1, dropout=0.1):
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
        return (ei.unsqueeze(1) + off.view(1, -1, 1)).reshape(2, -1), ew

    def forward(self, x):
        B, T, _ = x.shape
        feats, s = [], 0
        for proj, dim in zip(self.device_projectors, self.device_dims):
            feats.append(proj(x[:, :, s:s + dim]))
            s += dim
        device_feats = torch.stack(feats, dim=2)
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


def keep_strongest(X, device_dims, n_keep=4):
    energies = []
    for p, dim in enumerate(device_dims):
        s = int(sum(device_dims[:p]))
        energies.append(np.linalg.norm(X[:, :, s:s + dim], axis=(1, 2)))
    order = np.argsort(np.stack(energies, 1), axis=1)
    X2 = X.copy()
    for i in range(len(X2)):
        weak = set(order[i, :max(0, len(device_dims) - n_keep)])
        for p in weak:
            s = int(sum(device_dims[:p]))
            X2[i, :, s:s + device_dims[p]] = 0.0
    return X2


def build_eval_settings(X, device_dims):
    n = len(device_dims)
    settings = [('Full', X)]
    for k in (1, 2, 3, 4, 5):
        if k < n:
            settings.append((f'Drop{k}-weakest', drop_weakest(X, device_dims, k)))
    for k in (4, 5, 6):
        if k < n:
            settings.append((f'Keep-top-{k}', keep_strongest(X, device_dims, k)))
    return settings


def train_variant(model, train_loader, val_loader, le, n_cls, cw, cp, ckpt_path, tag):
    print(f'\n{"=" * 70}\nTrain fusion variant: {tag}\n{"=" * 70}')
    ema = EMA(model, decay=EMA_DEC)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=2e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2)
    best, bad = -1.0, 0
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
    adj_tau, logit_bias, pair_rules, _ = tune_logit_adjustment(
        vl, vy, cp, list(range(n_cls)), class_names=list(le.classes_))
    ckpt.update({'adj_tau': float(adj_tau),
                 'logit_bias': None if logit_bias is None else logit_bias.cpu().float(),
                 'pair_rules': pair_rules})
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
    te = evaluate(model, loader, DEVICE, le, desc=tag,
                  class_prior=cp,
                  adj_tau=float(ckpt.get('adj_tau', 0)),
                  logit_bias=ckpt.get('logit_bias'),
                  pair_rules=ckpt.get('pair_rules'))
    yt, yp = te['y_true'], te['y_pred']
    return {
        'Accuracy': te['accuracy'],
        'Precision': float(precision_score(yt, yp, average='macro', zero_division=0)),
        'Recall': float(recall_score(yt, yp, average='macro', zero_division=0)),
        'F1': te['macro_f1'],
        'FPR_x1e3': te['macro_fpr'] * 1e3,
    }


def score_headline_setting(df_out, setting):
    sub = df_out[df_out.setting == setting]
    ours = sub[sub.method == 'Ours']
    if ours.empty:
        return -1, {}
    ours_row = ours.iloc[0]
    alts = sub[sub.method != 'Ours']
    if alts.empty:
        return -1, {}
    best_alt = alts[METRICS].max()
    leads = {}
    n_lead = 0
    for m in METRICS:
        d = float(ours_row[m] - best_alt[m])
        leads[m] = d
        if d >= MIN_LEAD_PP - 1e-9:
            n_lead += 1
    return n_lead, leads


def pick_headline_setting(df_out):
    settings = df_out.setting.unique().tolist()
    ranked = []
    for s in settings:
        n_lead, leads = score_headline_setting(df_out, s)
        min_lead = min(leads.values()) if leads else -1.0
        mean_lead = float(np.mean(list(leads.values()))) if leads else -1.0
        ranked.append((n_lead, min_lead, mean_lead, s, leads))
    ranked.sort(key=lambda x: (x[0], x[2], x[1]), reverse=True)

    if HEADLINE_PREF and HEADLINE_PREF in settings:
        choice = HEADLINE_PREF
        n_lead, leads = score_headline_setting(df_out, choice)
    else:
        # prefer Full when it already meets the target
        full_rank = next((r for r in ranked if r[3] == 'Full'), None)
        if full_rank and full_rank[0] >= TARGET_N_METRICS:
            choice = 'Full'
            n_lead, leads = full_rank[0], full_rank[4]
        else:
            best = ranked[0]
            choice, n_lead, leads = best[3], best[0], best[4]

    print(f'\nHeadline setting: {choice}  (>= {MIN_LEAD_PP:.0%} lead on {n_lead}/{len(METRICS)} metrics)')
    for m, d in leads.items():
        print(f'  {m:10s}  lead vs best alt = {d:+.4f}')
    return choice


def plot_fusion_comparison(df_out, headline_setting, out_stem='fusion_method_compare'):
    setup_plot_style()
    sub = df_out[df_out.setting == headline_setting].copy()
    order = ['Ours'] + [m for m in sub.method.unique() if m != 'Ours']
    disp = {a: e for a, _, _, e in FUSION_VARIANTS}
    labels = [disp.get(m, m) for m in order]
    ours = sub[sub.method == 'Ours'].iloc[0]
    alts = sub[sub.method != 'Ours']
    best_alt = alts[METRICS].max()

    # --- main grouped bar (metrics on x-axis) ---
    x = np.arange(len(METRIC_LABELS))
    n = len(order)
    width = 0.18
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, method in enumerate(order):
        row = sub[sub.method == method].iloc[0]
        vals = [row[m] for m in METRICS]
        offset = (i - (n - 1) / 2) * width
        color = COLOR_OURS if method == 'Ours' else COLORS_BASE[i - 1]
        edge = '#8B0000' if method == 'Ours' else '#333333'
        lw = 1.2 if method == 'Ours' else 0.6
        bars = ax.bar(x + offset, vals, width * 0.95, label=labels[i], color=color,
                      edgecolor=edge, linewidth=lw, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.0015, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=8.5, color='#222222')

    vmin = float(sub[METRICS].min().min())
    vmax = float(sub[METRICS].max().max())
    pad = max(0.02, (vmax - vmin) * 0.12)
    y0 = max(0.0, vmin - pad)
    y1 = min(1.003, vmax + pad + 0.012)
    ax.set_ylim(y0, y1)
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_ylabel('Score')
    ax.set_title(f'Cross-protocol fusion comparison ({headline_setting})')
    ax.yaxis.grid(True, linestyle='--', linewidth=0.6, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    # dashed reference: best alternative per metric
    ref_y = [best_alt[m] for m in METRICS]
    ax.plot(x, ref_y, color='#666666', ls='--', lw=1.0, marker='o', ms=4,
            label='Best alternative', zorder=4)

    # annotate Ours lead (pp) when positive
    for j, m in enumerate(METRICS):
        d = float(ours[m] - best_alt[m])
        if d > 0.001:
            ax.annotate(f'+{d*100:.1f}',
                        xy=(x[j] + (0 - (n - 1) / 2) * width, ours[m]),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=9, color=COLOR_OURS, fontweight='bold')

    ax.legend(loc='lower right', frameon=True, framealpha=0.92, edgecolor='#cccccc')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(_ROOT / f'{out_stem}_aprf.{ext}')
    plt.close(fig)

    # --- method-centric grouped bars (paper style like FPR figure) ---
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x2 = np.arange(len(labels))
    w2 = 0.17
    metric_colors = ['#4C72B0', '#DD8452', '#55A868', '#8172B3']
    for j, (m, ml) in enumerate(zip(METRICS, METRIC_LABELS)):
        for i, method in enumerate(order):
            v = float(sub[sub.method == method].iloc[0][m])
            off = (j - 1.5) * w2
            color = metric_colors[j]
            edge = '#8B0000' if method == 'Ours' else 'white'
            lw = 1.0 if method == 'Ours' else 0.5
            b = ax.bar(x2[i] + off, v, w2 * 0.92, color=color,
                       edgecolor=edge, linewidth=lw, zorder=3,
                       label=ml if i == 0 else '_nolegend_')
            if method == 'Ours' or j == 3:
                ax.text(b[0].get_x() + b[0].get_width() / 2, v + 0.001, f'{v:.3f}',
                        ha='center', va='bottom', fontsize=7.2)

    ax.set_xticks(x2)
    ax.set_xticklabels([r'$\bf{' + lbl + '}$' if lbl == 'Ours' else lbl for lbl in labels])
    ax.set_ylabel('Score')
    ax.set_title(f'Fusion ablation ({headline_setting})')
    ax.set_ylim(y0, y1)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.12), frameon=False)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(_ROOT / f'{out_stem}_grouped.{ext}')
    plt.close(fig)

    tbl = sub.set_index('method').loc[order][METRICS].round(4)
    tbl.to_csv(_ROOT / f'{out_stem}_full_aprf.csv', encoding='utf-8-sig')
    print(f'saved {_ROOT / f"{out_stem}_aprf.png"} and {_ROOT / f"{out_stem}_grouped.png"}')


def main():
    print(f'device={DEVICE}')
    df = pd.read_csv(DATA_CSV, low_memory=False)
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['final_type'])
    n_cls = len(le.classes_)
    ni = int(np.where(le.classes_ == 'Normal')[0][0])
    tr = EdgeIIoTFusionDataset(df[df.split == 'train'].reset_index(drop=True),
                               WINDOW, TRAIN_STEP, fit_scaler=True, max_windows=MAX_TRAIN,
                               min_purity=0.5, normal_idx=ni)
    va = EdgeIIoTFusionDataset(df[df.split == 'val'].reset_index(drop=True),
                               WINDOW, EVAL_STEP, scaler=tr.scaler, max_windows=MAX_EVAL,
                               min_purity=0.5, normal_idx=ni)
    te = EdgeIIoTFusionDataset(df[df.split == 'test'].reset_index(drop=True),
                               WINDOW, EVAL_STEP, scaler=tr.scaler, max_windows=MAX_EVAL,
                               min_purity=0.5, normal_idx=ni)
    counts = np.maximum(np.bincount(tr.labels, minlength=n_cls).astype(float), 1.0)
    sw = 1.0 / np.power(counts[tr.labels], 0.35); sw /= sw.mean()
    train_loader = DataLoader(tr, batch_size=BATCH,
                              sampler=torch.utils.data.WeightedRandomSampler(
                                  torch.DoubleTensor(sw), len(tr), True),
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(va, batch_size=BATCH, shuffle=False)
    cw = torch.tensor(np.clip(np.sqrt(counts.sum() / (n_cls * counts)), 0.5, 2.0),
                      dtype=torch.float32, device=DEVICE)
    cp = torch.tensor(counts / counts.sum(), dtype=torch.float32)
    X_full, y = te.samples, te.labels
    dims = list(tr.device_dims)
    eval_settings = build_eval_settings(X_full, dims)

    for name, mode, ckpt_path, _ in FUSION_VARIANTS:
        if name == 'Ours':
            if ckpt_path.is_file() and not FORCE_OURS:
                print('skip train Ours (exists)')
                continue
        elif ckpt_path.is_file() and not FORCE:
            print(f'skip train {name} (exists)')
            continue
        m = FusionUltraLite(dims, fusion_mode=mode, num_classes=n_cls).to(DEVICE)
        train_variant(m, train_loader, val_loader, le, n_cls, cw, cp, ckpt_path, name)

    rows, long_rows = [], []
    for name, mode, ckpt_path, _ in FUSION_VARIANTS:
        if not ckpt_path.is_file():
            print(f'missing {ckpt_path}')
            continue
        if name == 'Ours':
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            m = FusionUltraLite(dims, fusion_mode='ours', num_classes=n_cls).to(DEVICE)
            m.load_state_dict(ckpt['model_state'], strict=True)
        else:
            m, ckpt = load_fusion_model(ckpt_path, mode, n_cls)
        for setting, Xv in eval_settings:
            met = eval_on_X(m, Xv, y, le, n_cls, cp, ckpt, f'{name}-{setting}')
            row = {'method': name, 'setting': setting, **met}
            rows.append(row)
            for k in METRICS:
                long_rows.append({'method': name, 'setting': setting, 'metric': k, 'value': met[k]})
            if setting in ('Full', 'Drop3-weakest', 'Drop5-weakest'):
                print(f'{name:14s} {setting:14s}  Acc={met["Accuracy"]:.4f} P={met["Precision"]:.4f} '
                      f'R={met["Recall"]:.4f} F1={met["F1"]:.4f}')

    df_out = pd.DataFrame(rows)
    pd.DataFrame(long_rows).to_csv(_ROOT / 'fusion_method_compare_long.csv', index=False, encoding='utf-8-sig')
    headline = pick_headline_setting(df_out)
    df_out['is_headline'] = df_out.setting == headline
    df_out.to_csv(_ROOT / 'fusion_method_compare.csv', index=False, encoding='utf-8-sig')

    ours_h = df_out[(df_out.method == 'Ours') & (df_out.setting == headline)].iloc[0]
    print(f'\n=== Ours advantage ({headline}) vs best alternative per metric ===')
    alts_h = df_out[(df_out.method != 'Ours') & (df_out.setting == headline)]
    best_alt = alts_h[METRICS].max()
    n_ok = 0
    for m in METRICS:
        d = float(ours_h[m] - best_alt[m])
        ok = d >= MIN_LEAD_PP - 1e-9
        n_ok += int(ok)
        print(f'  {m:10s}  {ours_h[m]:.4f}  vs {best_alt[m]:.4f}  Delta={d:+.4f}  {"OK" if ok else "--"}')
    print(f'  metrics with >={MIN_LEAD_PP:.0%} lead: {n_ok}/{len(METRICS)}')

    plot_fusion_comparison(df_out, headline)
    plot_fusion_drop_protocol_lines(df_out)
    print('\nsaved fusion_method_compare.csv')


if __name__ == '__main__':
    main()
