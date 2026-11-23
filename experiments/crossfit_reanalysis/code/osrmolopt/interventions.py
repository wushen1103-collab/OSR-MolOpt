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


def censor_tasks(tasks: list[dict], floor: float | None) -> list[dict]:
    result = copy.deepcopy(tasks)
    if floor is None:
        return result
    for task in result:
        for row in task['cal'] + task['evaluation']:
            p1t, p2t, p1o, p2o = [max(value, floor) for value in row['potencies']]
            row['delta_target'] = p2t - p1t
            row['delta_margin'] = p2t - p2o - (p1t - p1o)
    return result

def ridge_result(task: dict) -> dict:
    pt, pm = models.local_predictions(task, task['evaluation'], 'ridge')
    return models.evaluate(models.select(task['evaluation'], pt, pm), task)

def hist_result(train: list[dict], held: dict) -> dict:
    pt, pm, _, yt, ym = models.fit_models(train, held, 'hist', 'vina_only')
    pt, pm = models.adapt(pt, pm, yt, ym, 'bias50')
    return models.evaluate(models.select(held['evaluation'], pt[6:], pm[6:]), held)

def run_level(tasks: list[dict], label: str, floor: float | None, original_values: np.ndarray) -> dict:
    rows = {key: [] for key in ('Vina-MO', 'Theil-Sen', 'Ridge', 'HIST-SAFE')}
    oracle = []
    for held in tasks:
        train = [task for task in tasks if task['task'] != held['task']]
        rows['Vina-MO'].append({'task': held['task'], **models.baseline(held, 'vina')})
        rows['Theil-Sen'].append({'task': held['task'], **models.baseline(held, 'theilsen')})
        rows['Ridge'].append({'task': held['task'], **ridge_result(held)})
        rows['HIST-SAFE'].append({'task': held['task'], **hist_result(train, held)})
        oracle.append(any((r['delta_target'] * r['delta_margin'] > 0 for r in held['evaluation'])))
    summaries = {method: models.summarize(values) for method, values in rows.items()}
    paired = {method: models.paired(values, rows['Vina-MO']) for method, values in rows.items() if method != 'Vina-MO'}
    return {'level': label, 'floor': floor, 'individual_measurement_censor_rate': 0.0 if floor is None else float(np.mean(original_values < floor)), 'oracle_feasible_tasks': int(sum(oracle)), 'oracle_feasible_rate': float(np.mean(oracle)), 'summaries': summaries, 'paired_vs_vina': paired, 'task_results': rows}
