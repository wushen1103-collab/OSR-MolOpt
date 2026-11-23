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


def make_row(row_id: str, feature: dict, label: dict | None) -> dict:
    return {'row_id': row_id, 'vina_target': float(feature['vina_delta_target']), 'vina_margin': float(feature['vina_delta_margin']), 'delta_target': 0.0 if label is None else float(label['delta_target']), 'delta_margin': 0.0 if label is None else float(label['delta_margin']), 'contact': {}}

def select_vina(rows: list[dict]) -> dict:
    pt = np.asarray([r['vina_target'] for r in rows], float) / models.medabs([r['vina_target'] for r in rows])
    pm = np.asarray([r['vina_margin'] for r in rows], float) / models.medabs([r['vina_margin'] for r in rows])
    return models.select(rows, pt, pm)

def freeze_one(historical: list[dict], held: dict) -> dict:
    all_rows = held['cal'] + held['evaluation']
    picks = {'Vina-MO': select_vina(held['evaluation'])}
    for name, family in (('six-shot Theil-Sen', 'theilsen'), ('six-shot Ridge', 'ridge')):
        pt, pm = models.local_predictions(held, held['evaluation'], family)
        picks[name] = models.select(held['evaluation'], pt, pm)
    hpt, hpm, _, hyt, hym = models.fit_models(historical, held, 'hist', 'vina_only')
    picks['cross-campaign Hist no residual'] = models.select(held['evaluation'], hpt[6:], hpm[6:])
    apt, apm = models.adapt(hpt, hpm, hyt, hym, 'bias50')
    picks['HIST-SAFE'] = models.select(held['evaluation'], apt[6:], apm[6:])
    rpt, rpm, _, _, _ = models.fit_models(historical, held, 'ridge', 'vina_only')
    picks['cross-campaign Ridge no residual'] = models.select(held['evaluation'], rpt[6:], rpm[6:])
    return picks
