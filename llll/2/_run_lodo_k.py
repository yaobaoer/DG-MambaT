# -*- coding: utf-8 -*-
"""
Leave-k-Devices-Out — IoT fusion weather (7 devices)
训练: 清零 k 个设备通道（遍历全部组合）
验证: Full-7 | 测试: Full-7
输出: lodo_leave{k}_device_metrics.csv, lodo_leave_k_device_metrics.csv
"""
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import _run_lodo as lodo

_ROOT = Path(r'E:\apt\YES\2')
K_VALUES = [3, 4]
CACHE_DIR = lodo.CACHE_DIR / 'lodo_k_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def combo_key(k, combo):
    names = [lodo.DEVICES[i] for i in combo]
    return f'k{k}_' + '+'.join(names)


def _cache_load_fold(pack, model_name):
    if f'{model_name}_f1' in pack:
        return {
            'acc': float(pack[f'{model_name}_acc']),
            'prec': float(pack[f'{model_name}_prec']),
            'rec': float(pack[f'{model_name}_rec']),
            'f1': float(pack[f'{model_name}_f1']),
        }
    return None


def run_fold_all_models(tr_ld, va_ld, te_ld, slices, n_cls, cw, dims, hold_idxs, fold_label):
    hold_list = list(hold_idxs)
    fold_scores = {}
    if fold_label:
        print(f'\n######## {fold_label} ########')

    for name in lodo.MODEL_NAMES:
        is_grid = name == 'GRID'
        t0 = time.time()
        model = lodo.build_model(name, dims, n_cls)
        model, _ = lodo.train_model(
            model, tr_ld, va_ld, slices, hold_list, n_cls, cw, is_grid=is_grid)
        sc = lodo.eval_loader(model, te_ld, slices, n_cls, hold_d=None, is_grid=is_grid)
        fold_scores[name] = sc
        print(f'  {name} Target-F1={sc["f1"]:.4f} ({time.time() - t0:.0f}s)')
        del model
        if lodo.DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    return fold_scores


def summarize(k, results, fold_keys, source):
    rows, metric_rows = [], []
    for m in lodo.MODEL_NAMES:
        f1_scores = np.array(
            [lodo._fold_metric_val(results[m][fk], 'f1') for fk in fold_keys], dtype=np.float64)
        src = source.get(m, float('nan'))
        rows.append({
            'model': m,
            'k': k,
            'n_folds': len(fold_keys),
            'Source_F1': src,
            'Target_mean': float(f1_scores.mean()),
            'Target_std': float(f1_scores.std(ddof=1)),
            'Target_worst': float(f1_scores.min()),
            'Gap_mean': float(src - f1_scores.mean()) if np.isfinite(src) else float('nan'),
            'Gap_worst': float(src - f1_scores.min()) if np.isfinite(src) else float('nan'),
        })
        mrow = {'model': m, 'k': k, 'n_folds': len(fold_keys)}
        has_full = all(
            isinstance(results[m][fk], dict) and 'acc' in results[m][fk] for fk in fold_keys)
        for mk, label in lodo.METRIC_KEYS:
            if has_full or mk == 'f1':
                vals = np.array(
                    [lodo._fold_metric_val(results[m][fk], mk) for fk in fold_keys], dtype=np.float64)
                mrow[f'{label}_mean'] = float(vals.mean())
                mrow[f'{label}_std'] = float(vals.std(ddof=1))
        metric_rows.append(mrow)

    df_metrics = pd.DataFrame(metric_rows).set_index('model')
    out_metrics = _ROOT / f'lodo_leave{k}_device_metrics.csv'
    out_csv = _ROOT / f'lodo_leave{k}_device_out.csv'
    out_txt = _ROOT / f'lodo_leave{k}_device_out.txt'
    df_sum = pd.DataFrame(rows).set_index('model')
    cols = ['Source_F1', 'Target_mean', 'Target_std', 'Target_worst', 'Gap_mean', 'Gap_worst']
    lines = [
        f'Leave-{k}-Devices-Out on IoT fusion weather',
        f'Train: zero {k} device channels per combo | Val/Test: full-7',
        f'n_combos={len(fold_keys)} epochs={lodo.EPOCHS} seed={lodo.SEED}',
        '',
        df_sum[cols].to_string(float_format=lambda x: f'{x:.6f}'),
    ]
    out_txt.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    df_sum.to_csv(out_csv)
    df_metrics.sort_values('F1_mean', ascending=False).to_csv(out_metrics)
    print(f'\n[k={k}] saved {out_metrics}')
    return df_metrics


def run_k(k, tr_ld, va_ld, te_ld, slices, n_cls, cw, dims, source):
    combos = list(itertools.combinations(range(len(lodo.DEVICES)), k))
    fold_keys = [combo_key(k, c) for c in combos]
    results = {m: {} for m in lodo.MODEL_NAMES}

    print('\n' + '=' * 88)
    print(f'Leave-{k}-Devices-Out: {len(combos)} combinations')
    print('=' * 88)

    for combo, fk in zip(combos, fold_keys):
        cache_p = CACHE_DIR / f'{fk}.npz'
        names = [lodo.DEVICES[i] for i in combo]
        if cache_p.is_file():
            pack = np.load(cache_p, allow_pickle=True)
            for m in lodo.MODEL_NAMES:
                met = _cache_load_fold(pack, m)
                if met is not None:
                    results[m][fk] = met
            print(f'[cache] {fk}')
            continue

        fold_scores = run_fold_all_models(
            tr_ld, va_ld, te_ld, slices, n_cls, cw, dims, combo,
            fold_label=f'hold-out devices = {names}')
        np.savez(cache_p, **lodo._cache_save_fold(fold_scores))
        for m, sc in fold_scores.items():
            results[m][fk] = sc

    return summarize(k, results, fold_keys, source)


def main():
    t0 = time.time()
    torch.manual_seed(lodo.SEED)
    np.random.seed(lodo.SEED)

    df = pd.read_csv(lodo.DATA_CSV, low_memory=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    valid = df['final_type'].value_counts()[df['final_type'].value_counts() > 10].index
    df = df[df['final_type'].isin(valid)].copy()
    le = __import__('sklearn.preprocessing', fromlist=['LabelEncoder']).LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['final_type'])
    n_cls = len(le.classes_)
    print('classes', list(le.classes_))

    df_tr, df_va, df_te = lodo.stratified_timegroup_split(df)
    print(f'rows train/val/test {len(df_tr)}/{len(df_va)}/{len(df_te)}')

    tr_ds = lodo.FusionWindows(df_tr, lodo.WINDOW, lodo.TRAIN_STEP, fit=True)
    va_ds = lodo.FusionWindows(df_va, lodo.WINDOW, lodo.EVAL_STEP, scaler=tr_ds.scaler)
    te_ds = lodo.FusionWindows(df_te, lodo.WINDOW, lodo.EVAL_STEP, scaler=tr_ds.scaler)
    slices = tr_ds.slices
    dims = tr_ds.device_dims
    counts = np.maximum(np.bincount(tr_ds.y.numpy(), minlength=n_cls).astype(np.float64), 1)
    sw = 1.0 / np.sqrt(counts[tr_ds.y.numpy()])
    sw = sw / sw.mean()
    tr_ld = DataLoader(tr_ds, lodo.BATCH, sampler=WeightedRandomSampler(
        torch.DoubleTensor(sw), len(tr_ds), True))
    va_ld = DataLoader(va_ds, lodo.BATCH, shuffle=False)
    te_ld = DataLoader(te_ds, lodo.BATCH, shuffle=False)
    cw = torch.tensor(
        np.clip(np.sqrt(counts.sum() / (n_cls * counts)), 0.5, 3.0),
        dtype=torch.float32, device=lodo.DEVICE)

    source = {m: lodo.parse_test_f1(_ROOT / fn) for m, fn in lodo.SOURCE_FILES.items()}
    print('Source F1:', {k: round(v, 4) for k, v in source.items()})

    all_metrics = []
    for k in K_VALUES:
        dm = run_k(k, tr_ld, va_ld, te_ld, slices, n_cls, cw, dims, source)
        all_metrics.append(dm.reset_index())

    # merge k=2 (if exists) with new runs
    parts = []
    for k in [2, 3, 4]:
        p = _ROOT / f'lodo_leave{k}_device_metrics.csv'
        if p.is_file():
            parts.append(pd.read_csv(p))
    if parts:
        combined = pd.concat(parts, ignore_index=True)
        combined_path = _ROOT / 'lodo_leave_k_device_metrics.csv'
        combined.to_csv(combined_path, index=False)
        print(f'\nCombined (k=2/3/4) -> {combined_path}')

    print(f'Total elapsed {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
