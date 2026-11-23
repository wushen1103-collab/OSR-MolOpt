"""Reconstruct archived endpoints, paired statistics and reporting-limit sets."""
import copy
import json
import math
from pathlib import Path
import numpy as np
from . import data_io, decoupling, reporting_limits


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9), (a, b)


def clipped(task, floor):
    task = copy.deepcopy(task)
    for row in task['cal'] + task['evaluation']:
        p = list(row['potencies']) if floor is None else [max(v, floor) for v in row['potencies']]
        row['delta_target'] = p[1] - p[0]
        row['delta_margin'] = p[1] - p[3] - (p[0] - p[2])
    return task


def run():
    tasks = {t['task']: t for t in data_io.historical()}
    archive = data_io.load('data/records/censoring_decoupling_experiment.json')
    checked, summaries = 0, []
    for experiment in archive['experiments']:
        all_picks = {}
        for condition, record in experiment['conditions'].items():
            floor = experiment['floor'] if record['evaluation_labels'] == 'censored' else None
            view = {key: clipped(task, floor) for key, task in tasks.items()}
            assert decoupling.oracle_count(list(view.values())) == record['oracle_feasible_tasks'] == 38
            all_picks[condition] = {}
            for method, outcomes in record['task_results'].items():
                recalculated = []
                choices = {}
                for outcome in outcomes:
                    task = view[outcome['task']]
                    row = next(r for r in task['evaluation'] if r['row_id'] == outcome['row_id'])
                    sign = outcome['orientation']
                    assert sign in [-1., 1.]
                    t, m = sign * row['delta_target'], sign * row['delta_margin']
                    close(t, outcome['target']); close(m, outcome['margin'])
                    assert (t > 0 and m > 0) == outcome['dual_success']
                    assert (t < 0 and m < 0) == outcome['severe']
                    recalculated.append(dict(outcome, target=t, margin=m))
                    choices[outcome['task']] = (row['row_id'], sign)
                    checked += 1
                summary = decoupling.summarize(recalculated)
                for key, value in summary.items():
                    close(value, record[method][key])
                all_picks[condition][method] = choices
            effect = decoupling.paired_effect(record['task_results']['HIST-SAFE'], record['task_results']['Vina-MO'])
            for key in ['difference_pp','exact_one_sided_p']:
                if key in effect and key in record['paired_hist_vs_vina']:
                    close(effect[key], record['paired_hist_vs_vina'][key])
            np.testing.assert_allclose(effect['bootstrap_95_ci_pp'], record['paired_hist_vs_vina']['bootstrap_95_ci_pp'], atol=1e-9)
        a, b, c, d = list(all_picks)
        assert all_picks[a]['HIST-SAFE'] == all_picks[c]['HIST-SAFE']
        assert all_picks[b]['HIST-SAFE'] == all_picks[d]['HIST-SAFE']
        assert all(all_picks[key]['Vina-MO'] == all_picks[a]['Vina-MO'] for key in all_picks)
        summaries.append({'level': experiment['level'], 'hist_counts': [r['HIST-SAFE']['successes'] for r in experiment['conditions'].values()]})
    external = {t['task']: t for t in data_io.davis(include_evaluation=True)}
    point_archive = data_io.load('data/records/davis_external_blind_results.json')
    bounds_archive = data_io.load('data/records/interval_censoring_audit.json')
    selection = data_io.load('data/protocols/davis_frozen_selections.json')['selections']
    interval_count, point_count = 0, 0
    for method, outcomes in point_archive['task_results'].items():
        expected = {r['task']: r for r in bounds_archive['method_results'][method]['records']}
        for outcome in outcomes:
            task = external[outcome['task']]
            pick = selection[task['task']][method]
            row = next(r for r in task['evaluation'] if r['row_id'] == pick['row_id'])
            sign = pick['orientation']
            target, margin = sign * row['delta_target'], sign * row['delta_margin']
            close(target, outcome['target_gain']); close(margin, outcome['margin_gain'])
            iv = [reporting_limits.potency_interval(v) for v in row['kd_nm']]
            target_set = reporting_limits.orient(reporting_limits.linear_interval([(1, iv[1]), (-1, iv[0])]), sign)
            margin_set = reporting_limits.orient(reporting_limits.linear_interval([(1, iv[1]), (-1, iv[3]), (-1, iv[0]), (1, iv[2])]), sign)
            for endpoint, actual in [('target', target_set), ('margin', margin_set)]:
                wanted = expected[task['task']][endpoint + '_interval']
                for x, y in zip(actual, wanted):
                    close(x, float(y))
                assert reporting_limits.classify(actual) == expected[task['task']][endpoint + '_status']
                interval_count += 1
            point_count += 1
    ceiling = sum(value == 10000 for task in external.values() for row in task['evaluation'] for value in row['kd_nm'])
    assert ceiling == 164 and point_count == 54 and interval_count == 108
    result = {'status':'pass', 'historical_tasks':39, 'original_potency_slots':1872,
              'decoupled_outcomes_recomputed':checked, 'experiments':summaries,
              'external_frozen_choices':point_count, 'external_endpoint_intervals':interval_count,
              'external_evaluation_slots':216, 'external_ceiling_slots':ceiling,
              'model_refitting':False, 'docking':False}
    assert checked == 624
    return result
