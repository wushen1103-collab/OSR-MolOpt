"""CPU-only, checkpointed refitting on the frozen tasks and candidate directions."""
import copy
import json
import os
from pathlib import Path
import time
import numpy as np
from . import data_io, models, interventions, decoupling, external_selection, independent


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def historical_pick(tasks, index):
    held = tasks[index]
    train = [task for task in tasks if task['task'] != held['task']]
    blind = decoupling.label_blind_selection_task(held)
    rows = blind['evaluation']
    out = {'Vina-MO': decoupling.vina_pick(blind)}
    for name, family in [('Theil-Sen','theilsen'),('Ridge','ridge')]:
        pt, pm = models.local_predictions(blind, rows, family)
        out[name] = models.select(rows, pt, pm)
    pt, pm, _, yt, ym = models.fit_models(train, blind, 'hist', 'vina_only')
    out['Global Hist'] = models.select(rows, pt[6:], pm[6:])
    apt, apm = models.adapt(pt, pm, yt, ym, 'bias50')
    out['HIST-SAFE'] = models.select(rows, apt[6:], apm[6:])
    pt, pm, _, _, _ = models.fit_models(train, blind, 'ridge', 'vina_only')
    out['Global Ridge'] = models.select(rows, pt[6:], pm[6:])
    return out


def checkpoint(path, fingerprint, compute, inject_fault=False):
    if path.exists():
        result = json.loads(path.read_text(encoding='utf-8'))
        if result['fingerprint'] == fingerprint:
            return result['result'], True
    result = compute()
    payload = {'fingerprint': fingerprint, 'result': result}
    if inject_fault:
        atomic(path.with_name(path.stem + '.uncommitted.json'), payload)
        raise RuntimeError('Injected failure before checkpoint commit')
    atomic(path, payload)
    return result, False


def smoke(output, inject_fault=False):
    tasks = data_io.historical()
    return checkpoint(output / 'smoke.json', data_io.data_fingerprint(),
                      lambda: historical_pick(tasks, 0), inject_fault=inject_fault)


def run(output, response_curve=False):
    tasks = data_io.historical()
    fingerprint = data_io.data_fingerprint()
    jobs = output / 'checkpoints'
    floors = [('original', None), ('floor5_9.9pct',5.),
              ('global_q25_24.9pct',float(np.quantile([p for t in tasks for r in t['cal'] + t['evaluation'] for p in r['potencies']], .25)))]
    if response_curve:
        record = data_io.load('data/records/censoring_intervention_study.json')
        floors += [(r['level'], r['floor']) for r in record['levels'][3:]]
    picks, resumed = {}, 0
    for name, floor in floors:
        view = interventions.censor_tasks(tasks, floor)
        picks[name] = {}
        for i, task in enumerate(view):
            path = jobs / f'{name}_{i:02d}.json'
            result, cached = checkpoint(path, fingerprint, lambda: historical_pick(view, i))
            resumed += cached
            picks[name][task['task']] = result
            if i % 10 == 0:
                print(f'{name}: {i+1}/{len(tasks)} folds complete', flush=True)
    archived = data_io.load('data/records/censoring_decoupling_experiment.json')
    reconstructed = []
    for experiment in archived['experiments']:
        level, floor = experiment['level'], experiment['floor']
        view = {t['task']: t for t in interventions.censor_tasks(tasks, floor)}
        results = {}
        for code, record in experiment['conditions'].items():
            training = 'original' if record['training_labels'] == 'raw' else level
            by_method = {'HIST-SAFE': [], 'Vina-MO': []}
            for task in tasks:
                evaluation = task if record['evaluation_labels'] == 'raw' else view[task['task']]
                for method in by_method:
                    choice = picks[training if method == 'HIST-SAFE' else 'original'][task['task']][method]
                    actual = decoupling.evaluate_pick(evaluation, choice)
                    wanted = next(r for r in record['task_results'][method] if r['task'] == task['task'])
                    assert actual['row_id'] == wanted['row_id'] and actual['orientation'] == wanted['orientation'], (level,code,task['task'],method)
                    np.testing.assert_allclose([actual['target'],actual['margin']], [wanted['target'],wanted['margin']], atol=1e-9)
                    by_method[method].append(dict(actual, task=task['task']))
            results[code] = {'task_results': by_method, **{method: decoupling.summarize(rows) for method,rows in by_method.items()},
                             'paired_hist_vs_vina': decoupling.paired_effect(by_method['HIST-SAFE'],by_method['Vina-MO'])}
        reconstructed.append({'level':level, 'conditions':results})
    external_picks = {}
    for i, task in enumerate(data_io.davis()):
        selected, cached = checkpoint(jobs / f'external_{i:02d}.json', fingerprint,
                                     lambda: external_selection.freeze_one(tasks, task))
        external_picks[task['task']] = selected
        resumed += cached
    frozen = data_io.load('data/protocols/davis_frozen_selections.json')['selections']
    for task, methods in external_picks.items():
        for method, actual in methods.items():
            wanted = frozen[task][method]
            assert (actual['row_id'],actual['orientation']) == (wanted['row_id'],wanted['orientation']), (task,method)
    independent_outcomes = {method: [] for method in independent.METHODS}
    for i, task in enumerate(t for t in tasks if t['cohort'] == 'confirmatory'):
        for method in independent.METHODS:
            choice = independent.choose(task['cal'], task['evaluation'], method)
            independent_outcomes[method].append(dict(decoupling.evaluate_pick(task,choice),task=task['task']))
    independent_counts = {method: sum(r['dual_success'] for r in rows) for method,rows in independent_outcomes.items()}
    assert independent_counts == {'sixshot_theilsen':10, 'sixshot_ridge':9, 'vina_mo':13}, independent_counts
    historical_counts = {}
    for method in ['HIST-SAFE','Vina-MO','Theil-Sen','Ridge','Global Hist','Global Ridge']:
        outcomes = [decoupling.evaluate_pick(task,picks['original'][task['task']][method]) for task in tasks]
        historical_counts[method] = sum(r['dual_success'] for r in outcomes)
    assert historical_counts == {'HIST-SAFE':28,'Vina-MO':20,'Theil-Sen':23,'Ridge':21,'Global Hist':25,'Global Ridge':22}, historical_counts
    payload = {'status':'pass','historical_counts':historical_counts,'independent_counts':independent_counts,
               'historical_tasks':39,'external_tasks':9,'decoupling':reconstructed,
               'external_frozen_choices_reproduced':54,'resumed_checkpoints':int(resumed),
               'docking_recomputed':False,'gpu_used':False,'response_curve_included':response_curve}
    atomic(output / 'refit_results.json',payload)
    print(json.dumps({key:value for key,value in payload.items() if key!='decoupling'}),flush=True)
    return payload
