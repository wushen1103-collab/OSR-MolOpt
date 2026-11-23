"""Numerical implementation extracted from the frozen experiment source."""
from __future__ import annotations
import copy, json, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import binomtest, spearmanr, beta
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import Ridge, TheilSenRegressor, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
SEED = 20261117

AA = ("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "MSE", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL")

def medabs(values) -> float:
    return max(float(np.median(np.abs(np.asarray(values, dtype=float)))), 1e-06)

def robust_center_scale(cal_values, values) -> np.ndarray:
    cal = np.asarray(cal_values, dtype=float)
    values = np.asarray(values, dtype=float)
    center = float(np.median(cal))
    scale = max(float(np.median(np.abs(cal - center))), float(np.std(cal)) * 0.25, 1e-06)
    return np.clip((values - center) / scale, -8.0, 8.0)

def safe_corr(x, y) -> float:
    x, y = (np.asarray(x, float), np.asarray(y, float))
    if len(x) < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    v = spearmanr(x, y).statistic
    return float(v) if np.isfinite(v) else 0.0

def local_predictions(task: dict, rows: list[dict], kind: str) -> tuple[np.ndarray, np.ndarray]:
    cal = task['cal']
    output = []
    for xkey, ykey in (('vina_target', 'delta_target'), ('vina_margin', 'delta_margin')):
        x = np.asarray([[r[xkey]] for r in cal], float)
        z = np.asarray([[r[xkey]] for r in rows], float)
        y = np.asarray([r[ykey] for r in cal], float) / medabs([r[ykey] for r in cal])
        model = TheilSenRegressor(random_state=SEED) if kind == 'theilsen' else Ridge(alpha=1.0, solver='lsqr')
        pipe = make_pipeline(StandardScaler(), model)
        pipe.fit(x, y)
        output.append(np.asarray(pipe.predict(z), float))
    return (output[0], output[1])

def task_arrays(task: dict, rows: list[dict], feature_set: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cal = task['cal']
    vt = robust_center_scale([r['vina_target'] for r in cal], [r['vina_target'] for r in rows])
    vm = robust_center_scale([r['vina_margin'] for r in cal], [r['vina_margin'] for r in rows])
    tht, thm = local_predictions(task, rows, 'theilsen')
    rt, rm = local_predictions(task, rows, 'ridge')
    cal_vt = [r['vina_target'] for r in cal]
    cal_vm = [r['vina_margin'] for r in cal]
    cal_yt = [r['delta_target'] for r in cal]
    cal_ym = [r['delta_margin'] for r in cal]
    meta = np.asarray([task['assay_ic50'], safe_corr(cal_vt, cal_yt), safe_corr(cal_vm, cal_ym), np.mean(np.sign(cal_vt) == np.sign(cal_yt)), np.mean(np.sign(cal_vm) == np.sign(cal_ym)), safe_corr(cal_yt, cal_ym)], float)
    base = np.column_stack([vt, vm, tht, thm, rt, rm, vt * vm, np.abs(vt), np.abs(vm), np.repeat(meta[None, :], len(rows), axis=0)])
    if feature_set == 'contact':
        contacts = np.asarray([[r['contact'][a] for a in AA] for r in rows], float)
        contacts = np.clip(contacts / 3.0, -3.0, 3.0)
        totals = np.column_stack([np.sum(np.abs(contacts), axis=1), np.sum(contacts > 0, axis=1), np.sum(contacts < 0, axis=1)])
        base = np.column_stack([base, contacts, totals, contacts * vt[:, None], contacts * vm[:, None]])
    yt = np.asarray([r['delta_target'] for r in rows], float) / medabs(cal_yt)
    ym = np.asarray([r['delta_margin'] for r in rows], float) / medabs(cal_ym)
    return (base, yt, ym)

def fit_models(train_tasks: list[dict], held: dict, family: str, feature_set: str):
    xs, yts, yms = ([], [], [])
    for task in train_tasks:
        x, yt, ym = task_arrays(task, task['cal'] + task['evaluation'], feature_set)
        xs.append(x)
        yts.append(yt)
        yms.append(ym)
    x = np.vstack(xs)
    yt = np.concatenate(yts)
    ym = np.concatenate(yms)
    if family == 'ridge':
        mt = make_pipeline(StandardScaler(), Ridge(alpha=10.0, solver='lsqr'))
        mm = make_pipeline(StandardScaler(), Ridge(alpha=10.0, solver='lsqr'))
    elif family == 'extra':
        kw = dict(n_estimators=128, max_depth=6, min_samples_leaf=4, max_features=0.7, random_state=SEED, n_jobs=1)
        mt, mm = (ExtraTreesRegressor(**kw), ExtraTreesRegressor(**kw))
    elif family == 'hist':
        kw = dict(max_iter=120, learning_rate=0.04, max_leaf_nodes=10, min_samples_leaf=15, l2_regularization=3.0, random_state=SEED)
        mt, mm = (HistGradientBoostingRegressor(**kw), HistGradientBoostingRegressor(**kw))
    else:
        raise ValueError(family)
    mt.fit(x, yt)
    mm.fit(x, ym)
    all_rows = held['cal'] + held['evaluation']
    xh, yht, yhm = task_arrays(held, all_rows, feature_set)
    pt, pm = (np.asarray(mt.predict(xh)), np.asarray(mm.predict(xh)))
    uncertainty = np.zeros((2, len(all_rows)), float)
    if family == 'extra':
        uncertainty[0] = np.std(np.asarray([e.predict(xh) for e in mt.estimators_]), axis=0)
        uncertainty[1] = np.std(np.asarray([e.predict(xh) for e in mm.estimators_]), axis=0)
    return (pt, pm, uncertainty, yht, yhm)

def adapt(pt, pm, true_t, true_m, mode: str):
    pt, pm = (pt.copy(), pm.copy())
    if mode in {'bias50', 'bias100'}:
        w = 0.5 if mode == 'bias50' else 1.0
        pt += w * float(np.median(true_t[:6] - pt[:6]))
        pm += w * float(np.median(true_m[:6] - pm[:6]))
    return (pt, pm)

def select(rows: list[dict], pt, pm, ut=None, um=None, lcb=0.0) -> dict:
    pt, pm = (np.asarray(pt), np.asarray(pm))
    ut = np.zeros_like(pt) if ut is None else np.asarray(ut)
    um = np.zeros_like(pm) if um is None else np.asarray(um)
    candidates = []
    for i, row in enumerate(rows):
        for sign in (-1.0, 1.0):
            st = sign * pt[i] - lcb * ut[i]
            sm = sign * pm[i] - lcb * um[i]
            candidates.append((min(st, sm), row['row_id'], sign, st, sm))
    best = max((x[0] for x in candidates))
    chosen = min((x for x in candidates if abs(x[0] - best) < 1e-12), key=lambda x: (x[1], x[2]))
    return {'row_id': chosen[1], 'orientation': chosen[2], 'selection_score': float(chosen[0]), 'predicted_target': float(chosen[3]), 'predicted_margin': float(chosen[4])}

def evaluate(pick: dict, task: dict) -> dict:
    row = next((r for r in task['evaluation'] if r['row_id'] == pick['row_id']))
    sign = float(pick['orientation'])
    target, margin = (sign * row['delta_target'], sign * row['delta_margin'])
    st = medabs([r['delta_target'] for r in task['cal']])
    sm = medabs([r['delta_margin'] for r in task['cal']])
    return {**pick, 'target': float(target), 'margin': float(margin), 'dual_success': bool(target > 0 and margin > 0), 'severe': bool(target < 0 and margin < 0), 'utility': float(min(target / st, margin / sm))}

def baseline(task: dict, kind: str) -> dict:
    rows = task['evaluation']
    if kind == 'vina':
        pt = np.asarray([r['vina_target'] for r in rows]) / medabs([r['vina_target'] for r in rows])
        pm = np.asarray([r['vina_margin'] for r in rows]) / medabs([r['vina_margin'] for r in rows])
    else:
        pt, pm = local_predictions(task, rows, 'theilsen')
        if kind == 'th75':
            vt = np.asarray([r['vina_target'] for r in rows]) / medabs([r['vina_target'] for r in rows])
            vm = np.asarray([r['vina_margin'] for r in rows]) / medabs([r['vina_margin'] for r in rows])
            pt, pm = (0.25 * vt + 0.75 * pt, 0.25 * vm + 0.75 * pm)
    return evaluate(select(rows, pt, pm), task)

def summarize(rows: list[dict]) -> dict:
    return {'task_count': len(rows), 'dual_success_count': int(sum((r['dual_success'] for r in rows))), 'dual_success_rate': float(np.mean([r['dual_success'] for r in rows])), 'mean_target_gain': float(np.mean([r['target'] for r in rows])), 'mean_margin_gain': float(np.mean([r['margin'] for r in rows])), 'severe_count': int(sum((r['severe'] for r in rows))), 'severe_rate': float(np.mean([r['severe'] for r in rows])), 'mean_utility': float(np.mean([r['utility'] for r in rows]))}

def paired(rows, vina) -> dict:
    wins = sum((a['dual_success'] and (not b['dual_success']) for a, b in zip(rows, vina)))
    losses = sum((b['dual_success'] and (not a['dual_success']) for a, b in zip(rows, vina)))
    p = float(binomtest(wins, wins + losses, 0.5, alternative='greater').pvalue) if wins + losses else 1.0
    return {'success_rate_difference': float(np.mean([r['dual_success'] for r in rows]) - np.mean([r['dual_success'] for r in vina])), 'discordant_wins': wins, 'discordant_losses': losses, 'exact_one_sided_p': p}
