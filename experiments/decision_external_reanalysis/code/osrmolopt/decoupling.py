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

from . import models, interventions


def label_blind_selection_task(task: dict) -> dict:
    result = copy.deepcopy(task)
    for row in result['evaluation']:
        row['delta_target'] = 0.0
        row['delta_margin'] = 0.0
    return result

def hist_pick(train: list[dict], held: dict) -> dict:
    blind = label_blind_selection_task(held)
    pt, pm, _, yt, ym = models.fit_models(train, blind, 'hist', 'vina_only')
    pt, pm = models.adapt(pt, pm, yt, ym, 'bias50')
    pick = models.select(blind['evaluation'], pt[6:], pm[6:])
    return pick

def vina_pick(task: dict) -> dict:
    rows = task['evaluation']
    pt = np.asarray([r['vina_target'] for r in rows], float) / models.medabs([r['vina_target'] for r in rows])
    pm = np.asarray([r['vina_margin'] for r in rows], float) / models.medabs([r['vina_margin'] for r in rows])
    return models.select(rows, pt, pm)

def hybrid_evaluation_task(raw: dict, censored: dict, evaluation_censored: bool) -> dict:
    result = copy.deepcopy(censored if evaluation_censored else raw)
    result['evaluation'] = copy.deepcopy(censored['evaluation'] if evaluation_censored else raw['evaluation'])
    return result

def evaluate_pick(task: dict, pick: dict) -> dict:
    row = next((r for r in task['evaluation'] if r['row_id'] == pick['row_id']))
    sign = float(pick['orientation'])
    target, margin = (sign * row['delta_target'], sign * row['delta_margin'])
    return {'row_id': pick['row_id'], 'orientation': sign, 'target': float(target), 'margin': float(margin), 'dual_success': bool(target > 0 and margin > 0), 'severe': bool(target < 0 and margin < 0)}

def summarize(rows: list[dict]) -> dict:
    return {'task_count': len(rows), 'successes': int(sum((r['dual_success'] for r in rows))), 'success_rate': float(np.mean([r['dual_success'] for r in rows])), 'mean_target': float(np.mean([r['target'] for r in rows])), 'mean_margin': float(np.mean([r['margin'] for r in rows])), 'severe': int(sum((r['severe'] for r in rows)))}

def paired_effect(hist: list[dict], vina: list[dict], draws: int=20000) -> dict:
    a = np.asarray([r['dual_success'] for r in hist], bool)
    b = np.asarray([r['dual_success'] for r in vina], bool)
    wins = int(np.sum(a & ~b))
    losses = int(np.sum(b & ~a))
    n = wins + losses
    exact = float(binomtest(wins, n, 0.5, alternative='greater').pvalue) if n else 1.0
    diff = a.astype(float) - b.astype(float)
    rng = np.random.default_rng(SEED)
    sampled = np.mean(diff[rng.integers(0, len(diff), size=(draws, len(diff)))], axis=1)
    return {'difference_pp': float(100 * np.mean(diff)), 'discordant_wins': wins, 'discordant_losses': losses, 'exact_one_sided_p': exact, 'bootstrap_95_ci_pp': [float(100 * np.quantile(sampled, 0.025)), float(100 * np.quantile(sampled, 0.975))], 'bootstrap_draws': draws}

def oracle_count(tasks: list[dict]) -> int:
    return int(sum((any((row['delta_target'] * row['delta_margin'] > 0 for row in task['evaluation'])) for task in tasks)))

def transition(a: list[dict], b: list[dict]) -> dict:
    aa = np.asarray([r['dual_success'] for r in a], bool)
    bb = np.asarray([r['dual_success'] for r in b], bool)
    lost = int(np.sum(aa & ~bb))
    gained = int(np.sum(~aa & bb))
    n = lost + gained
    return {'lost_successes': lost, 'gained_successes': gained, 'exact_one_sided_p_lost_greater': float(binomtest(lost, n, 0.5, alternative='greater').pvalue) if n else 1.0}

def run_floor(raw_tasks: list[dict], floor: float, level: str) -> dict:
    censored_tasks = interventions.censor_tasks(raw_tasks, floor)
    raw_map = {task['task']: task for task in raw_tasks}
    cens_map = {task['task']: task for task in censored_tasks}
    selections = {'raw': {}, 'censored': {}, 'vina': {}}
    for held_raw in raw_tasks:
        name = held_raw['task']
        held_cens = cens_map[name]
        raw_train = [task for task in raw_tasks if task['task'] != name]
        cens_train = [task for task in censored_tasks if task['task'] != name]
        selections['raw'][name] = hist_pick(raw_train, held_raw)
        selections['censored'][name] = hist_pick(cens_train, held_cens)
        selections['vina'][name] = vina_pick(held_raw)
    conditions = {}
    specs = {'A_raw_train_raw_eval': ('raw', False), f'B_{level}_train_raw_eval': ('censored', False), f'C_raw_train_{level}_eval': ('raw', True), f'D_{level}_train_{level}_eval': ('censored', True)}
    for condition, (training_state, eval_censored) in specs.items():
        hist_rows, vina_rows = ([], [])
        for name in [task['task'] for task in raw_tasks]:
            eval_task = hybrid_evaluation_task(raw_map[name], cens_map[name], eval_censored)
            hist_rows.append({'task': name, **evaluate_pick(eval_task, selections[training_state][name])})
            vina_rows.append({'task': name, **evaluate_pick(eval_task, selections['vina'][name])})
        conditions[condition] = {'training_labels': 'censored' if training_state == 'censored' else 'raw', 'evaluation_labels': 'censored' if eval_censored else 'raw', 'HIST-SAFE': summarize(hist_rows), 'Vina-MO': summarize(vina_rows), 'paired_hist_vs_vina': paired_effect(hist_rows, vina_rows), 'oracle_feasible_tasks': oracle_count(censored_tasks if eval_censored else raw_tasks), 'task_results': {'HIST-SAFE': hist_rows, 'Vina-MO': vina_rows}}
    raw_picks, cens_picks = (selections['raw'], selections['censored'])
    changed = [name for name in raw_picks if (raw_picks[name]['row_id'], raw_picks[name]['orientation']) != (cens_picks[name]['row_id'], cens_picks[name]['orientation'])]
    row_changed = [name for name in raw_picks if raw_picks[name]['row_id'] != cens_picks[name]['row_id']]
    orientation_only = [name for name in changed if name not in row_changed]
    a = conditions['A_raw_train_raw_eval']['task_results']['HIST-SAFE']
    b = conditions[f'B_{level}_train_raw_eval']['task_results']['HIST-SAFE']
    return {'level': level, 'floor': floor, 'conditions': conditions, 'selection_change': {'changed_candidate_or_orientation_tasks': len(changed), 'changed_row_tasks': len(row_changed), 'orientation_only_change_tasks': len(orientation_only), 'changed_tasks': changed}, 'A_to_B_success_transition': transition(a, b)}
