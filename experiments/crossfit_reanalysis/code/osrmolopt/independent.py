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

METHODS = ("sixshot_theilsen", "sixshot_ridge", "vina_mo")

def fit_predict(cal: list[dict], test: list[dict], xkey: str, ykey: str, kind: str) -> np.ndarray:
    x = np.asarray([[r[xkey]] for r in cal], dtype=float)
    z = np.asarray([[r[xkey]] for r in test], dtype=float)
    y = np.asarray([r[ykey] for r in cal], dtype=float)
    scaler = StandardScaler()
    xs, zs = (scaler.fit_transform(x), scaler.transform(z))
    if kind == 'ridge':
        model = Ridge(alpha=1.0, solver='lsqr')
    elif kind == 'theilsen':
        model = TheilSenRegressor(random_state=20261117)
    else:
        raise ValueError(kind)
    model.fit(xs, y)
    return np.asarray(model.predict(zs), dtype=float)

def choose(cal: list[dict], test: list[dict], method: str) -> dict:
    if method == 'vina_mo':
        pt = np.asarray([r['vina_target'] for r in test])
        pm = np.asarray([r['vina_margin'] for r in test])
    else:
        kind = 'theilsen' if method == 'sixshot_theilsen' else 'ridge'
        pt = fit_predict(cal, test, 'vina_target', 'delta_target', kind)
        pm = fit_predict(cal, test, 'vina_margin', 'delta_margin', kind)
    ts = max(float(np.median(np.abs(pt))), 1e-08)
    ms = max(float(np.median(np.abs(pm))), 1e-08)
    ranked = []
    for index, row in enumerate(test):
        for sign in (-1.0, 1.0):
            ranked.append((min(sign * pt[index] / ts, sign * pm[index] / ms), row['row_id'], sign, float(pt[index]), float(pm[index])))
    best = max((x[0] for x in ranked))
    score, rid, sign, pred_t, pred_m = min((x for x in ranked if abs(x[0] - best) <= 1e-15), key=lambda x: (x[1], x[2]))
    return {'row_id': rid, 'orientation': sign, 'selection_score': score, 'predicted_target': sign * pred_t, 'predicted_margin': sign * pred_m}

def ci(k: int, n: int) -> list[float]:
    return [0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))]

def summarize(rows: list[dict]) -> dict:
    target = np.asarray([r['oriented_target_gain'] for r in rows], dtype=float)
    margin = np.asarray([r['oriented_margin_gain'] for r in rows], dtype=float)
    success = (target > 0) & (margin > 0)
    severe = (target < 0) & (margin < 0)
    return {'task_count': len(rows), 'dual_success_count': int(success.sum()), 'dual_success_rate': float(success.mean()), 'dual_success_ci95': ci(int(success.sum()), len(rows)), 'target_success_rate': float(np.mean(target > 0)), 'margin_success_rate': float(np.mean(margin > 0)), 'mean_target_gain': float(target.mean()), 'std_target_gain': float(target.std(ddof=1)), 'mean_margin_gain': float(margin.mean()), 'std_margin_gain': float(margin.std(ddof=1)), 'median_target_gain': float(np.median(target)), 'median_margin_gain': float(np.median(margin)), 'severe_dual_degradation_count': int(severe.sum()), 'severe_dual_degradation_rate': float(severe.mean())}

def compare(primary: list[dict], other: list[dict], seed: int) -> dict:
    a = {x['task']: x for x in primary}
    b = {x['task']: x for x in other}
    tasks = sorted(a)
    sa = np.asarray([a[t]['dual_success'] for t in tasks], dtype=bool)
    sb = np.asarray([b[t]['dual_success'] for t in tasks], dtype=bool)
    wins = int(np.sum(sa & ~sb))
    losses = int(np.sum(~sa & sb))
    discordant = wins + losses
    utility = np.asarray([a[t]['realized_robust_utility'] - b[t]['realized_robust_utility'] for t in tasks])
    non_ties = utility[np.abs(utility) > 1e-12]
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(tasks), size=(20000, len(tasks)))
    boot = (sa.astype(float)[indexes] - sb.astype(float)[indexes]).mean(axis=1)
    return {'dual_success_rate_difference': float(sa.mean() - sb.mean()), 'bootstrap_difference_ci95': [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))], 'discordant_wins': wins, 'discordant_losses': losses, 'exact_one_sided_mcnemar_p': 1.0 if discordant == 0 else float(binom.sf(wins - 1, discordant, 0.5)), 'mean_utility_difference': float(utility.mean()), 'median_utility_difference': float(np.median(utility)), 'utility_positive_differences': int(np.sum(non_ties > 0)), 'utility_non_ties': int(len(non_ties)), 'exact_one_sided_utility_sign_p': 1.0 if len(non_ties) == 0 else float(binom.sf(np.sum(non_ties > 0) - 1, len(non_ties), 0.5))}
