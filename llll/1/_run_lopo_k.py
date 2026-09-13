# -*- coding: utf-8 -*-
"""
Leave-k-Protocols-Out (k=2,3,4) — X-IIoTID fusion
训练/验证: 清零 k 个协议通道（遍历所有组合）
测试: 全协议特征
输出: lopo_leave{k}_protocol_metrics.csv + lopo_leave_k_protocol_metrics.csv
"""
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

import _run_lopo as lopo

_ROOT = Path(r'E:\apt\YES\1')
K_VALUES = [2]  # 仅 Leave-2-Protocols-Out；k=3/k=4 已取消
CACHE_DIR = lopo.CACHE_DIR / 'lopo_k_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def combo_key(k, combo):
    names = [lopo.PROTOCOL_GROUPS[i] for i in combo]
    return f'k{k}_' + '+'.join(names)


def load_source_f1():
    source = {}
    pred_map = {
        'CKAN-BiLSTM': 'ckan_test_pred.npz',
        'GRID': 'grid_test_pred.npz',
        'WaveMamba': 'wavemamba_test_pred.npz',
        'Transformer-IDS': 'transform_ids_test_pred.npz',
        'FeCo': 'feco_test_pred.npz',
        'MPGNN': 'mpgnn_test_pred.npz',
        'TCG-IDS': 'tcg_ids_test_pred.npz',
        'Ours': 'ultralite_test_pred.npz',
    }
    for name, fn in pred_map.items():
        p = lopo._TMP / fn
        if p.is_file():
            pack = np.load(p)
            source[name] = float(f1_score(
                pack['y_true'], pack['y_pred'], average='macro', zero_division=0))
        else:
            source[name] = float('nan')
    return source


def prepare_data():
    torch.manual_seed(lopo.SEED)
    np.random.seed(lopo.SEED)
    ckpt0 = torch.load(lopo.CKPT_OURS, map_location='cpu', weights_only=False)
    classes = list(ckpt0['classes'])
    n_cls = len(classes)
    df = pd.read_csv(lopo.DATA_CSV, low_memory=False)
    df['label_encoded'] = df['final_type'].map({c: i for i, c in enumerate(classes)}).astype(np.int64)
    normal_idx = classes.index('Normal')
    df_train = df[df.split == 'train'].reset_index(drop=True)
    df_test = df[df.split == 'test'].reset_index(drop=True)
    train_ds = lopo.EdgeIIoTFusionDataset(
        df_train, lopo.WINDOW, lopo.TRAIN_STEP, fit_scaler=True,
        max_windows=lopo.MAX_TRAIN, min_purity=lopo.MIN_PURITY, normal_idx=normal_idx)
    test_ds = lopo.EdgeIIoTFusionDataset(
        df_test, lopo.WINDOW, lopo.EVAL_STEP, scaler=train_ds.scaler,
        max_windows=lopo.MAX_EVAL, min_purity=lopo.MIN_PURITY, normal_idx=normal_idx)
    X_all_tr, y_all_tr = train_ds.samples, train_ds.labels
    X_te, y_te = test_ds.samples, test_ds.labels
    device_dims = list(train_ds.device_dims)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=lopo.SEED)
    tr_idx, va_idx = next(sss.split(np.zeros(len(y_all_tr)), y_all_tr))
    X_tr0, y_tr = X_all_tr[tr_idx], y_all_tr[tr_idx]
    X_va0, y_va = X_all_tr[va_idx], y_all_tr[va_idx]
    print(f'device={lopo.DEVICE} train={len(y_tr)} val={len(y_va)} test={len(y_te)} dims={device_dims}')
    return classes, n_cls, device_dims, X_tr0, X_va0, X_te, y_tr, y_va, y_te


def summarize_k(k, results, fold_keys, source, model_names):
    rows = []
    metric_rows = []
    for m in model_names:
        f1_scores = np.array(
            [lopo._fold_metric_val(results[m][fk], 'f1') for fk in fold_keys], dtype=np.float64)
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
        for mk, label in lopo.METRIC_KEYS:
            if has_full or mk == 'f1':
                mu, sd = lopo._agg_fold(results, m, fold_keys, mk)
                mrow[f'{label}_mean'] = mu
                mrow[f'{label}_std'] = sd
        metric_rows.append(mrow)

    df_sum = pd.DataFrame(rows).set_index('model')
    df_metrics = pd.DataFrame(metric_rows).set_index('model')
    out_csv = _ROOT / f'lopo_leave{k}_protocol_out.csv'
    out_metrics = _ROOT / f'lopo_leave{k}_protocol_metrics.csv'
    out_txt = _ROOT / f'lopo_leave{k}_protocol_out.txt'
    cols = ['Source_F1', 'Target_mean', 'Target_std', 'Target_worst', 'Gap_mean', 'Gap_worst']
    lines = [
        f'Leave-{k}-Protocols-Out on X-IIoTID fusion',
        f'Train/val: zero {k} protocol channels per combo | Test: full features',
        f'n_combos={len(fold_keys)} epochs={lopo.EPOCHS} seed={lopo.SEED}',
        '',
        df_sum[cols].to_string(float_format=lambda x: f'{x:.6f}'),
    ]
    out_txt.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    df_sum.to_csv(out_csv)
    df_metrics.sort_values('F1_mean', ascending=False).to_csv(out_metrics)
    print(f'\n[k={k}] saved {out_metrics}')
    return df_metrics


def run_k(k, classes, n_cls, device_dims, X_tr0, X_va0, X_te, y_tr, y_va, y_te, source):
    model_names = [
        'Ours',
        'CKAN-BiLSTM', 'WaveMamba', 'Transformer-IDS', 'FeCo',
        'GRID', 'MPGNN', 'TCG-IDS',
    ]
    combos = list(itertools.combinations(range(len(lopo.PROTOCOL_GROUPS)), k))
    fold_keys = [combo_key(k, c) for c in combos]
    results = {m: {} for m in model_names}
    print(f'\n{"=" * 88}')
    print(f'Leave-{k}-Protocols-Out: {len(combos)} combinations')
    print('=' * 88)

    for combo, fk in zip(combos, fold_keys):
        cache_p = CACHE_DIR / f'{fk}.npz'
        names = [lopo.PROTOCOL_GROUPS[i] for i in combo]
        if cache_p.is_file():
            pack = np.load(cache_p, allow_pickle=True)
            for m in model_names:
                met = lopo._cache_load_fold(pack, m)
                if met is not None:
                    results[m][fk] = met
            print(f'[cache] {fk}')
            continue

        X_tr = lopo.zero_protocols(X_tr0, device_dims, combo)
        X_va = lopo.zero_protocols(X_va0, device_dims, combo)
        fold_scores = lopo.run_fold_all_models(
            X_tr, y_tr, X_va, y_va, X_te, y_te, device_dims, n_cls, classes,
            fold_label=f'hold-out protocols = {names}')
        np.savez(cache_p, **lopo._cache_save_fold(fold_scores))
        for m, sc in fold_scores.items():
            results[m][fk] = sc

    return summarize_k(k, results, fold_keys, source, model_names)


def main():
    t0 = time.time()
    source = load_source_f1()
    print('Source F1:', {k: round(v, 4) for k, v in source.items()})
    pack = prepare_data()
    classes, n_cls, device_dims, X_tr0, X_va0, X_te, y_tr, y_va, y_te = pack

    all_metrics = []
    for k in K_VALUES:
        dm = run_k(k, classes, n_cls, device_dims, X_tr0, X_va0, X_te, y_tr, y_va, y_te, source)
        all_metrics.append(dm.reset_index())

    combined = pd.concat(all_metrics, ignore_index=True)
    combined_path = _ROOT / 'lopo_leave_k_protocol_metrics.csv'
    combined.to_csv(combined_path, index=False)
    print(f'\nCombined metrics -> {combined_path}')
    print(f'Total elapsed {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
