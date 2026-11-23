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



def summarize(rows: list[dict]) -> dict:
    return {'task_count': len(rows), 'dual_success_count': int(sum((row['dual_success'] for row in rows))), 'dual_success_rate': float(np.mean([row['dual_success'] for row in rows])), 'mean_target_gain': float(np.mean([row['target_gain'] for row in rows])), 'mean_margin_gain': float(np.mean([row['margin_gain'] for row in rows])), 'severe_count': int(sum((row['severe_dual_degradation'] for row in rows))), 'severe_rate': float(np.mean([row['severe_dual_degradation'] for row in rows]))}

def paired(a: list[dict], b: list[dict]) -> dict:
    by_a, by_b = ({x['task']: x for x in a}, {x['task']: x for x in b})
    wins = sum((by_a[k]['dual_success'] and (not by_b[k]['dual_success']) for k in by_a))
    losses = sum((by_b[k]['dual_success'] and (not by_a[k]['dual_success']) for k in by_a))
    n = wins + losses
    p = float(binomtest(wins, n, p=0.5, alternative='greater').pvalue) if n else 1.0
    return {'wins': int(wins), 'losses': int(losses), 'discordant': int(n), 'exact_one_sided_p': p}
