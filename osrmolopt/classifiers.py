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


def logit(p):
    p = np.clip(np.asarray(p, float), 0.0001, 1 - 0.0001)
    return np.log(p / (1 - p))

def sigmoid(x):
    x = np.clip(np.asarray(x, float), -20, 20)
    return 1.0 / (1.0 + np.exp(-x))

def fit_classifier(train_tasks, held, family: str, feature_set: str):
    xs, yt, ym = ([], [], [])
    for task in train_tasks:
        rows = task['cal'] + task['evaluation']
        x, ytr, ymr = models.task_arrays(task, rows, feature_set)
        xs.append(x)
        yt.append(ytr > 0)
        ym.append(ymr > 0)
    x = np.vstack(xs)
    yt = np.concatenate(yt)
    ym = np.concatenate(ym)
    if family == 'hist':
        kw = dict(max_iter=160, learning_rate=0.035, max_leaf_nodes=9, min_samples_leaf=18, l2_regularization=5.0, random_state=SEED)
        mt, mm = (HistGradientBoostingClassifier(**kw), HistGradientBoostingClassifier(**kw))
    elif family == 'extra':
        kw = dict(n_estimators=192, max_depth=6, min_samples_leaf=5, max_features=0.7, class_weight='balanced', random_state=SEED, n_jobs=1)
        mt, mm = (ExtraTreesClassifier(**kw), ExtraTreesClassifier(**kw))
    else:
        raise ValueError(family)
    mt.fit(x, yt)
    mm.fit(x, ym)
    rows = held['cal'] + held['evaluation']
    xh, yht, yhm = models.task_arrays(held, rows, feature_set)
    return (mt.predict_proba(xh)[:, 1], mm.predict_proba(xh)[:, 1], yht, yhm)

def calibrate_prob(p, y, weight: float):
    observed = (np.asarray(y[:6]) > 0).astype(float)
    empirical_logit = float(np.log((observed.sum() + 0.5) / (len(observed) - observed.sum() + 0.5)))
    offset = empirical_logit - float(np.mean(logit(p[:6])))
    return sigmoid(logit(p) + weight * np.clip(offset, -2.0, 2.0))

def vina_prob(task, rows):
    vt = np.asarray([r['vina_target'] for r in rows], float) / models.medabs([r['vina_target'] for r in task['cal']])
    vm = np.asarray([r['vina_margin'] for r in rows], float) / models.medabs([r['vina_margin'] for r in task['cal']])
    return (sigmoid(vt), sigmoid(vm))

def select_prob(rows, pt, pm):
    candidates = []
    for i, row in enumerate(rows):
        candidates.append((min(pt[i], pm[i]), row['row_id'], 1.0, pt[i], pm[i]))
        candidates.append((min(1 - pt[i], 1 - pm[i]), row['row_id'], -1.0, 1 - pt[i], 1 - pm[i]))
    best = max((x[0] for x in candidates))
    x = min((x for x in candidates if abs(x[0] - best) < 1e-12), key=lambda z: (z[1], z[2]))
    return {'row_id': x[1], 'orientation': x[2], 'selection_score': float(x[0]), 'predicted_target': float(x[3]), 'predicted_margin': float(x[4])}
