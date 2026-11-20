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

from . import models


def task_meta(task: dict) -> np.ndarray:
    cal, ev = (task['cal'], task['evaluation'])
    cvt = np.asarray([r['vina_target'] for r in cal], float)
    cvm = np.asarray([r['vina_margin'] for r in cal], float)
    cyt = np.asarray([r['delta_target'] for r in cal], float)
    cym = np.asarray([r['delta_margin'] for r in cal], float)
    evt = np.asarray([r['vina_target'] for r in ev], float)
    evm = np.asarray([r['vina_margin'] for r in ev], float)
    return np.asarray([task['assay_ic50'], models.safe_corr(cvt, cyt), models.safe_corr(cvm, cym), models.safe_corr(cyt, cym), np.mean(np.sign(cvt) == np.sign(cyt)), np.mean(np.sign(cvm) == np.sign(cym)), np.mean(np.isclose(cyt, 0.0)), np.mean(np.isclose(cym, 0.0)), np.std(cyt), np.std(cym), len(np.unique(np.round(cyt, 8))) / 6.0, len(np.unique(np.round(cym, 8))) / 6.0, np.median(np.abs(evt)) / models.medabs(cvt), np.median(np.abs(evm)) / models.medabs(cvm)], float)

def candidate_arrays(task: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, float]]]:
    cal, rows = (task['cal'], task['evaluation'])
    vt = np.asarray([r['vina_target'] for r in rows], float) / models.medabs([r['vina_target'] for r in rows])
    vm = np.asarray([r['vina_margin'] for r in rows], float) / models.medabs([r['vina_margin'] for r in rows])
    tht, thm = models.local_predictions(task, rows, 'theilsen')
    rt, rm = models.local_predictions(task, rows, 'ridge')
    meta = task_meta(task)
    x, yt, ym, ids = ([], [], [], [])
    for i, row in enumerate(rows):
        for sign in (-1.0, 1.0):
            values = np.asarray([sign * vt[i], sign * vm[i], sign * tht[i], sign * thm[i], sign * rt[i], sign * rm[i]], float)
            x.append(np.concatenate([values, [min(values[0], values[1]), min(values[2], values[3]), min(values[4], values[5]), values[0] * values[1], np.std(values[[0, 2, 4]]), np.std(values[[1, 3, 5]])], meta]))
            yt.append(float(sign * row['delta_target'] > 0))
            ym.append(float(sign * row['delta_margin'] > 0))
            ids.append((row['row_id'], sign))
    return (np.asarray(x), np.asarray(yt), np.asarray(ym), ids)

def vina_pick(task: dict) -> tuple[str, float]:
    rows = task['evaluation']
    vt = np.asarray([r['vina_target'] for r in rows]) / models.medabs([r['vina_target'] for r in rows])
    vm = np.asarray([r['vina_margin'] for r in rows]) / models.medabs([r['vina_margin'] for r in rows])
    pick = models.select(rows, vt, vm)
    return (pick['row_id'], float(pick['orientation']))

def outcome(task: dict, pick: tuple[str, float]) -> dict:
    row = next((r for r in task['evaluation'] if r['row_id'] == pick[0]))
    target, margin = (pick[1] * row['delta_target'], pick[1] * row['delta_margin'])
    return {'task': task['task'], 'cohort': task['cohort'], 'row_id': pick[0], 'orientation': pick[1], 'target': float(target), 'margin': float(margin), 'dual_success': bool(target > 0 and margin > 0), 'severe': bool(target < 0 and margin < 0)}

def fit_predict(train: list[dict], held: dict, family: str) -> tuple[np.ndarray, list[tuple[str, float]]]:
    arrays = [candidate_arrays(task) for task in train]
    x = np.vstack([a[0] for a in arrays])
    yt = np.concatenate([a[1] for a in arrays])
    ym = np.concatenate([a[2] for a in arrays])
    weights = []
    for task in train:
        for row in task['evaluation']:
            w = 0.25 if abs(row['delta_target']) < 1e-12 or abs(row['delta_margin']) < 1e-12 else 1.0
            weights.extend([w, w])
    weights = np.asarray(weights, float)
    if family == 'hist':
        kw = dict(max_iter=120, learning_rate=0.04, max_leaf_nodes=8, min_samples_leaf=18, l2_regularization=4.0, random_state=SEED)
        mt, mm = (HistGradientBoostingClassifier(**kw), HistGradientBoostingClassifier(**kw))
        mt.fit(x, yt, sample_weight=weights)
        mm.fit(x, ym, sample_weight=weights)
    else:
        mt = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight='balanced', max_iter=2000, random_state=SEED))
        mm = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight='balanced', max_iter=2000, random_state=SEED))
        mt.fit(x, yt, logisticregression__sample_weight=weights)
        mm.fit(x, ym, logisticregression__sample_weight=weights)
    xh, _, _, ids = candidate_arrays(held)
    pt = mt.predict_proba(xh)[:, 1]
    pm = mm.predict_proba(xh)[:, 1]
    return (np.minimum(pt, pm), ids)

def ood_score(train: list[dict], held: dict) -> tuple[float, float]:
    x = np.asarray([task_meta(t) for t in train], float)
    h = task_meta(held)
    center = np.median(x, axis=0)
    scale = np.maximum(np.median(np.abs(x - center), axis=0), np.std(x, axis=0) * 0.25)
    active = scale > 1e-08
    z_train = np.max(np.abs((x[:, active] - center[active]) / scale[active]), axis=1)
    z_held = float(np.max(np.abs((h[active] - center[active]) / scale[active])))
    return (z_held, float(np.quantile(z_train, 0.95)))

def select_score(scores: np.ndarray, ids: list[tuple[str, float]]) -> tuple[str, float]:
    best = float(np.max(scores))
    return min((ids[i] for i in range(len(ids)) if abs(float(scores[i]) - best) < 1e-12), key=lambda value: (value[0], value[1]))

def evaluate_fold(train: list[dict], held: dict) -> dict[str, dict]:
    vp = vina_pick(held)
    out = {'Vina-MO': outcome(held, vp)}
    for family, label in (('logit', 'CENSOR-LOGIT'), ('hist', 'CENSOR-RANK')):
        scores, ids = fit_predict(train, held, family)
        pick = select_score(scores, ids)
        out[label] = outcome(held, pick)
        vina_index = ids.index(vp)
        advantage = float(np.max(scores) - scores[vina_index])
        z, z95 = ood_score(train, held)
        guarded = pick if advantage >= 0.1 and z <= z95 else vp
        g = outcome(held, guarded)
        g.update({'model_advantage': advantage, 'ood_score': z, 'train_ood_q95': z95, 'fallback': guarded == vp})
        out[label + '-GUARD'] = g
    return out

def summary(rows: list[dict]) -> dict:
    return {'tasks': len(rows), 'successes': int(sum((r['dual_success'] for r in rows))), 'success_rate': float(np.mean([r['dual_success'] for r in rows])), 'severe': int(sum((r['severe'] for r in rows))), 'severe_rate': float(np.mean([r['severe'] for r in rows])), 'mean_target': float(np.mean([r['target'] for r in rows])), 'mean_margin': float(np.mean([r['margin'] for r in rows])), 'fallback_rate': float(np.mean([r.get('fallback', False) for r in rows])) if rows else 0.0}

def paired(a: list[dict], b: list[dict]) -> dict:
    aa, bb = ({x['task']: x for x in a}, {x['task']: x for x in b})
    wins = sum((aa[k]['dual_success'] and (not bb[k]['dual_success']) for k in aa))
    losses = sum((bb[k]['dual_success'] and (not aa[k]['dual_success']) for k in aa))
    n = wins + losses
    return {'wins': wins, 'losses': losses, 'p_one_sided': float(binomtest(wins, n, 0.5, alternative='greater').pvalue) if n else 1.0}

def run_scheme(name: str, tasks: list[dict], train_for) -> dict:
    rows: dict[str, list[dict]] = {}
    for held in tasks:
        train = train_for(held)
        result = evaluate_fold(train, held)
        for method, row in result.items():
            rows.setdefault(method, []).append(row)
    return {'scheme': name, 'summaries': {k: summary(v) for k, v in rows.items()}, 'paired_vs_vina': {k: paired(v, rows['Vina-MO']) for k, v in rows.items() if k != 'Vina-MO'}, 'task_results': rows}
