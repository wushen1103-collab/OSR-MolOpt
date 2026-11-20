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
from . import gates as gate_models


def utility(task: dict, row: dict) -> float:
    st = models.medabs([r['delta_target'] for r in task['cal']])
    sm = models.medabs([r['delta_margin'] for r in task['cal']])
    return float(min(row['target'] / st, row['margin'] / sm))

def gate_feature(task: dict, base: dict[str, dict]) -> np.ndarray:
    diagnostic = base['CENSOR-RANK-GUARD']
    ratio = diagnostic['ood_score'] / max(diagnostic['train_ood_q95'], 1e-06)
    return np.concatenate([gate_models.task_meta(task), [diagnostic['model_advantage'], ratio]])

def crossfit_gate_training(universe: list[dict], excluded_cohort: str | None) -> tuple[np.ndarray, np.ndarray]:
    allowed = [t for t in universe if t['cohort'] != excluded_cohort]
    xs, ys = ([], [])
    for cohort in sorted({t['cohort'] for t in allowed}):
        inner_train = [t for t in allowed if t['cohort'] != cohort]
        for task in [t for t in allowed if t['cohort'] == cohort]:
            base = gate_models.evaluate_fold(inner_train, task)
            xs.append(gate_feature(task, base))
            ys.append(utility(task, base['CENSOR-RANK']) - utility(task, base['Vina-MO']))
    return (np.asarray(xs, float), np.asarray(ys, float))

def fit_gates(x: np.ndarray, y: np.ndarray):
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    ridge.fit(x, y)
    hist = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.04, max_leaf_nodes=5, min_samples_leaf=6, l2_regularization=5.0, random_state=SEED)
    hist.fit(x, y)
    return {'META-RIDGE': ridge, 'META-HIST': hist}

def choose(task: dict, base: dict[str, dict], model, threshold: float) -> dict:
    predicted = float(model.predict(gate_feature(task, base)[None, :])[0])
    use_rank = predicted > threshold
    row = dict(base['CENSOR-RANK'] if use_rank else base['Vina-MO'])
    row.update({'predicted_rank_utility_advantage': predicted, 'fallback': not use_rank, 'gate_threshold': threshold})
    return row

def evaluate_historical_nested(historical: list[dict]) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = defaultdict(list)
    for held_cohort in sorted({t['cohort'] for t in historical}):
        outer_train = [t for t in historical if t['cohort'] != held_cohort]
        x, y = crossfit_gate_training(historical, held_cohort)
        gates = fit_gates(x, y)
        for task in [t for t in historical if t['cohort'] == held_cohort]:
            base = gate_models.evaluate_fold(outer_train, task)
            output['Vina-MO'].append(base['Vina-MO'])
            output['CENSOR-RANK'].append(base['CENSOR-RANK'])
            for name, model in gates.items():
                output[name].append(choose(task, base, model, 0.0))
                output[name + '-CONSERVATIVE'].append(choose(task, base, model, 0.25))
    return output

def evaluate_davis_transfer(historical: list[dict], davis: list[dict]) -> dict[str, list[dict]]:
    x, y = crossfit_gate_training(historical, None)
    gates = fit_gates(x, y)
    output: dict[str, list[dict]] = defaultdict(list)
    for task in davis:
        base = gate_models.evaluate_fold(historical, task)
        output['Vina-MO'].append(base['Vina-MO'])
        output['CENSOR-RANK'].append(base['CENSOR-RANK'])
        for name, model in gates.items():
            output[name].append(choose(task, base, model, 0.0))
            output[name + '-CONSERVATIVE'].append(choose(task, base, model, 0.25))
    return output

def group_summary(rows: dict[str, list[dict]]) -> dict:
    result = {'summaries': {m: gate_models.summary(v) for m, v in rows.items()}, 'paired_vs_vina': {m: gate_models.paired(v, rows['Vina-MO']) for m, v in rows.items() if m != 'Vina-MO'}, 'task_results': rows}
    cohort = {}
    for method, values in rows.items():
        groups = defaultdict(list)
        for row in values:
            groups[row['cohort']].append(row)
        cohort[method] = {key: gate_models.summary(value) for key, value in sorted(groups.items())}
    result['cohort_summaries'] = cohort
    return result
