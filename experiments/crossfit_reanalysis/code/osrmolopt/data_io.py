"""Portable data access with separate selection and evaluation views."""
from pathlib import Path
import copy
import hashlib
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def historical():
    tasks = load('data/historical_tasks.json')['tasks']
    assert len(tasks) == 39 and all(len(t['cal']) == len(t['evaluation']) == 6 for t in tasks)
    return tasks


def davis(include_evaluation=False):
    tasks = load('data/davis/selection_tasks.json')['tasks']
    assert len(tasks) == 9
    if not include_evaluation:
        assert all(r['delta_target'] == r['delta_margin'] == 0 for t in tasks for r in t['evaluation'])
        return tasks
    with np.load(ROOT / 'data/davis/affinities.npz', allow_pickle=False) as arrays:
        matrix = arrays['kd_nm']
    index = load('data/davis/matrix_index.json')
    compounds = {key: i for i, key in enumerate(index['compound_ids'])}
    targets = {key: i for i, key in enumerate(index['target_ids'])}
    pairs = load('data/davis/compound_pairs.json')
    for task in tasks:
        _, target, off_target = task['task'].split(':')
        for i, row in enumerate(task['cal'] + task['evaluation']):
            pair = pairs[i]
            a, b = str(pair['compound_a']), str(pair['compound_b'])
            kd = [float(matrix[compounds[a], targets[target]]), float(matrix[compounds[b], targets[target]]),
                  float(matrix[compounds[a], targets[off_target]]), float(matrix[compounds[b], targets[off_target]])]
            p = [9 - math.log10(value) for value in kd]
            row.update(kd_nm=kd, potencies=p, delta_target=p[1]-p[0], delta_margin=p[1]-p[3]-(p[0]-p[2]))
    return tasks


def data_fingerprint():
    checksum = hashlib.sha256()
    for name in ['data/historical_tasks.json', 'data/davis/selection_tasks.json']:
        checksum.update((ROOT / name).read_bytes())
    for path in sorted((ROOT / 'osrmolopt').glob('*.py')):
        checksum.update(path.read_bytes())
    return checksum.hexdigest()
