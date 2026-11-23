"""Decision-consequence analysis and corrected Davis re-selection.

This program is deliberately CPU-only.  It reuses existing docking features and
never reads Davis evaluation labels while fitting or selecting candidates.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from scipy.stats import binomtest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "inputs"
OUTPUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "code"))
from osrmolopt import decoupling, models, reporting_limits  # noqa: E402

SEED = 20261117
N_BOOT = 20000


def load(name: str):
    return json.loads((INPUT / name).read_text(encoding="utf-8"))


def atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint() -> str:
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for path in sorted(INPUT.glob("*")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    for path in sorted((ROOT / "code" / "osrmolopt").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def crossfit_features(task: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-fit every donor calibration row; evaluation rows use all six shots."""
    x, yt, ym = models.task_arrays(task, task["cal"] + task["evaluation"], "vina_only")
    for i, row in enumerate(task["cal"]):
        five = copy.deepcopy(task)
        five["cal"] = [r for j, r in enumerate(task["cal"]) if j != i]
        x[i] = models.task_arrays(five, [row], "vina_only")[0][0]
    return x, yt, ym


def davis_views() -> tuple[list[dict], list[dict]]:
    """Return label-blind selection tasks and a separate labelled evaluation view."""
    blind = load("davis_selection_tasks.json")["tasks"]
    assert len(blind) == 9
    assert all(
        r["delta_target"] == 0 and r["delta_margin"] == 0
        for task in blind for r in task["evaluation"]
    )
    labelled = copy.deepcopy(blind)
    with np.load(INPUT / "affinities.npz", allow_pickle=False) as arrays:
        matrix = arrays["kd_nm"]
    index = load("matrix_index.json")
    compounds = {key: i for i, key in enumerate(index["compound_ids"])}
    targets = {key: i for i, key in enumerate(index["target_ids"])}
    pairs = load("compound_pairs.json")
    for task in labelled:
        _, target, off_target = task["task"].split(":")
        for i, row in enumerate(task["cal"] + task["evaluation"]):
            pair = pairs[i]
            a, b = str(pair["compound_a"]), str(pair["compound_b"])
            kd = [
                float(matrix[compounds[a], targets[target]]),
                float(matrix[compounds[b], targets[target]]),
                float(matrix[compounds[a], targets[off_target]]),
                float(matrix[compounds[b], targets[off_target]]),
            ]
            p = [9.0 - math.log10(value) for value in kd]
            row.update(
                kd_nm=kd,
                potencies=p,
                delta_target=p[1] - p[0],
                delta_margin=p[1] - p[3] - (p[0] - p[2]),
            )
    return blind, labelled


def fit_external(donors: list[dict], held: dict, cache: dict) -> dict:
    """Fit the corrected global models on all strict donors and select blindly."""
    rows = held["evaluation"]
    xh, yt, ym = models.task_arrays(held, held["cal"] + rows, "vina_only")
    x = np.vstack([cache[t["task"]][0] for t in donors])
    targets = [np.concatenate([cache[t["task"]][k] for t in donors]) for k in (1, 2)]
    choices = {"Vina-MO": decoupling.vina_pick(held)}
    for label, family in (("Theil-Sen", "theilsen"), ("Ridge", "ridge")):
        pt, pm = models.local_predictions(held, rows, family)
        choices[label] = models.select(rows, pt, pm)
    for family in ("hist", "ridge"):
        predictions = []
        for y in targets:
            if family == "hist":
                estimator = HistGradientBoostingRegressor(
                    max_iter=120,
                    learning_rate=0.04,
                    max_leaf_nodes=10,
                    min_samples_leaf=15,
                    l2_regularization=3.0,
                    random_state=SEED,
                )
            else:
                estimator = make_pipeline(StandardScaler(), Ridge(alpha=10.0, solver="lsqr"))
            estimator.fit(x, y)
            predictions.append(estimator.predict(xh))
        pt, pm = predictions
        if family == "hist":
            choices["Global Hist"] = models.select(rows, pt[6:], pm[6:])
            apt, apm = models.adapt(pt, pm, yt, ym, "bias50")
            choices["HIST-SAFE"] = models.select(rows, apt[6:], apm[6:])
        else:
            choices["Global Ridge"] = models.select(rows, pt[6:], pm[6:])
    return choices


def evaluate_interval(task: dict, pick: dict) -> dict:
    point = decoupling.evaluate_pick(task, pick)
    row = next(r for r in task["evaluation"] if r["row_id"] == pick["row_id"])
    iv = [reporting_limits.potency_interval(float(v)) for v in row["kd_nm"]]
    sign = float(pick["orientation"])
    target = reporting_limits.orient(
        reporting_limits.linear_interval([(1.0, iv[1]), (-1.0, iv[0])]), sign
    )
    margin = reporting_limits.orient(
        reporting_limits.linear_interval(
            [(1.0, iv[1]), (-1.0, iv[3]), (-1.0, iv[0]), (1.0, iv[2])]
        ),
        sign,
    )
    ts, ms = reporting_limits.classify(target), reporting_limits.classify(margin)
    return {
        **point,
        "target_interval": [reporting_limits.json_float(v) for v in target],
        "margin_interval": [reporting_limits.json_float(v) for v in margin],
        "target_status": ts,
        "margin_status": ms,
        "endpoint_status": "positive" if ts == ms == "positive" else (
            "negative" if "negative" in (ts, ms) else "unresolved"
        ),
        "fully_identified": all(math.isfinite(v) for v in target + margin),
    }


def checkpoint(path: Path, fp: str, compute, fault: bool = False):
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("fingerprint") == fp:
            return record["result"], True
    result = compute()
    payload = {"fingerprint": fp, "result": result}
    if fault:
        atomic(path.with_name(path.stem + ".uncommitted.json"), payload)
        raise RuntimeError("Intentional failure before checkpoint commit")
    atomic(path, payload)
    return result, False


def bootstrap_ci(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(arr), size=(N_BOOT, len(arr)))
    return np.quantile(arr[indices].mean(axis=1), [0.025, 0.975]).tolist()


def sign_test(values: list[float], alternative: str) -> dict:
    arr = np.asarray(values, dtype=float)
    pos, neg = int(np.sum(arr > 0)), int(np.sum(arr < 0))
    if alternative == "greater":
        p = binomtest(pos, pos + neg, 0.5, alternative="greater").pvalue if pos + neg else 1.0
    else:
        p = binomtest(neg, pos + neg, 0.5, alternative="greater").pvalue if pos + neg else 1.0
    return {"positive": pos, "negative": neg, "ties": int(np.sum(arr == 0)), "one_sided_p": float(p)}


def decision_analysis() -> dict:
    tasks = load("tasks_primary.json")
    archived = load("crossfit_results.json")
    view = archived["views"]["primary21_crossfit"]
    floor = float(archived["q25_floor"])
    a_picks = view["picks"]["original"]
    b_picks = view["picks"]["q25"]
    records = []
    for task in tasks:
        name = task["task"]
        pa, pb = a_picks[name]["HIST-SAFE"], b_picks[name]["HIST-SAFE"]
        ea, eb = decoupling.evaluate_pick(task, pa), decoupling.evaluate_pick(task, pb)
        st = models.medabs([r["delta_target"] for r in task["cal"]])
        sm = models.medabs([r["delta_margin"] for r in task["cal"]])
        candidates = []
        for row in task["evaluation"]:
            for sign in (-1.0, 1.0):
                target = sign * float(row["delta_target"])
                margin = sign * float(row["delta_margin"])
                candidates.append(min(target / st, margin / sm))
        oracle_utility = max(candidates)
        ua = min(float(ea["target"]) / st, float(ea["margin"]) / sm)
        ub = min(float(eb["target"]) / st, float(eb["margin"]) / sm)
        ra, rb = oracle_utility - ua, oracle_utility - ub
        if pa["row_id"] == pb["row_id"] and pa["orientation"] == pb["orientation"]:
            change = "unchanged"
        elif pa["row_id"] == pb["row_id"]:
            change = "orientation_reversal"
        else:
            change = "different_pair"
        dt, dm = eb["target"] - ea["target"], eb["margin"] - ea["margin"]
        if dt >= 0 and dm >= 0 and (dt > 0 or dm > 0):
            relation = "B_dominates_A"
        elif dt <= 0 and dm <= 0 and (dt < 0 or dm < 0):
            relation = "A_dominates_B"
        elif dt == 0 and dm == 0:
            relation = "same_outcome"
        else:
            relation = "tradeoff"
        records.append({
            "task": name,
            "change_type": change,
            "pareto_relation": relation,
            "A_row_id": pa["row_id"],
            "A_orientation": pa["orientation"],
            "B_row_id": pb["row_id"],
            "B_orientation": pb["orientation"],
            "A_target": ea["target"],
            "B_target": eb["target"],
            "B_minus_A_target": dt,
            "A_margin": ea["margin"],
            "B_margin": eb["margin"],
            "B_minus_A_margin": dm,
            "A_dual_success": ea["dual_success"],
            "B_dual_success": eb["dual_success"],
            "A_utility": ua,
            "B_utility": ub,
            "oracle_utility": oracle_utility,
            "A_regret": ra,
            "B_regret": rb,
            "B_minus_A_regret": rb - ra,
        })
    changed = [r for r in records if r["change_type"] != "unchanged"]

    def describe(rows: list[dict]) -> dict:
        output = {"task_count": len(rows)}
        for key in ("B_minus_A_target", "B_minus_A_margin", "B_minus_A_regret"):
            values = [float(r[key]) for r in rows]
            output[key] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "paired_bootstrap_95_ci_mean": bootstrap_ci(values),
                "sign_test_greater": sign_test(values, "greater"),
                "sign_test_less": sign_test(values, "less"),
            }
        return output

    return {
        "status": "complete",
        "q25_floor": floor,
        "task_count": len(records),
        "choice_change_counts": {key: sum(r["change_type"] == key for r in records) for key in (
            "unchanged", "orientation_reversal", "different_pair"
        )},
        "pareto_counts": {key: sum(r["pareto_relation"] == key for r in records) for key in (
            "B_dominates_A", "A_dominates_B", "same_outcome", "tradeoff"
        )},
        "all_tasks": describe(records),
        "changed_choices_only": describe(changed),
        "records": records,
    }


def external_one(index: int, fault: bool = False):
    donors = load("tasks_primary.json")
    blind, _ = davis_views()
    cache = {task["task"]: crossfit_features(task) for task in donors}
    held = blind[index]
    # Direct selection-label firewall check.
    before = fit_external(donors, held, cache)
    perturbed = copy.deepcopy(held)
    for row in perturbed["evaluation"]:
        row["delta_target"] = 1e6
        row["delta_margin"] = -1e6
    after = fit_external(donors, perturbed, cache)
    assert before == after
    return checkpoint(OUTPUT / "checkpoints" / f"davis_{index:02d}.json", fingerprint(), lambda: before, fault)


def external_run() -> dict:
    donors = load("tasks_primary.json")
    blind, labelled = davis_views()
    cache = {task["task"]: crossfit_features(task) for task in donors}
    fp, reused, selections = fingerprint(), 0, {}
    for i, task in enumerate(blind):
        choices, hit = checkpoint(
            OUTPUT / "checkpoints" / f"davis_{i:02d}.json",
            fp,
            lambda i=i: fit_external(donors, blind[i], cache),
        )
        reused += int(hit)
        selections[task["task"]] = choices
        print(json.dumps({"task": task["task"], "complete": i + 1, "total": len(blind)}), flush=True)
    methods = list(next(iter(selections.values())).keys())
    results = {method: [] for method in methods}
    for task in labelled:
        for method in methods:
            results[method].append({"task": task["task"], **evaluate_interval(task, selections[task["task"]][method])})
    summaries = {}
    for method, rows in results.items():
        summaries[method] = {
            "plugin_dual_success": int(sum(r["dual_success"] for r in rows)),
            "plugin_severe": int(sum(r["severe"] for r in rows)),
            "mean_plugin_target": float(np.mean([r["target"] for r in rows])),
            "mean_plugin_margin": float(np.mean([r["margin"] for r in rows])),
            "certified_positive": int(sum(r["endpoint_status"] == "positive" for r in rows)),
            "certified_negative": int(sum(r["endpoint_status"] == "negative" for r in rows)),
            "unresolved": int(sum(r["endpoint_status"] == "unresolved" for r in rows)),
        }
    historical = load("davis_frozen_selections.json")["selections"]
    historical_names = {
        "Vina-MO": "Vina-MO",
        "Theil-Sen": "six-shot Theil-Sen",
        "Ridge": "six-shot Ridge",
        "Global Hist": "cross-campaign Hist no residual",
        "HIST-SAFE": "HIST-SAFE",
        "Global Ridge": "cross-campaign Ridge no residual",
    }
    changed = {
        method: int(sum(
            (selections[task][method]["row_id"], selections[task][method]["orientation"])
            != (
                historical[task][historical_names[method]]["row_id"],
                historical[task][historical_names[method]]["orientation"],
            )
            for task in selections
        )) for method in methods
    }
    ceiling = sum(value == 10000.0 for task in labelled for row in task["evaluation"] for value in row["kd_nm"])
    output = {
        "status": "complete_retrospective_corrected_external_reselection",
        "fingerprint": fp,
        "donor_tasks": len(donors),
        "external_tasks": len(blind),
        "selection_reads_evaluation_labels": False,
        "historical_labels_were_previously_revealed": True,
        "docking_recomputed": False,
        "GPU": False,
        "cpu_threads": 1,
        "resumed_checkpoints": reused,
        "evaluation_slots_at_10000nM": ceiling,
        "evaluation_slot_count": len(labelled) * 6 * 4,
        "changed_vs_historical_frozen_selection": changed,
        "summaries": summaries,
        "selections": selections,
        "task_results": results,
    }
    atomic(OUTPUT / "davis_corrected_reselection.json", output)
    return output


def bounded(mode: str, timeout: int, extra: list[str] | None = None):
    command = [sys.executable, str(Path(__file__).resolve()), mode] + (extra or [])
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        out, _ = process.communicate(timeout=timeout)
        return process.returncode, out
    except subprocess.TimeoutExpired:
        process.kill()
        out, _ = process.communicate()
        return 124, out


def audit() -> dict:
    check = OUTPUT / "resource_checks"
    if check.exists():
        for path in check.glob("*"):
            if path.is_file():
                path.unlink()
    committed = OUTPUT / "checkpoints" / "davis_00.json"
    if committed.exists():
        committed.unlink()
    code, out = bounded("_one", 120, ["--fault"])
    assert code not in (0, 124) and not committed.exists(), out
    code, out = bounded("_one", 120)
    assert code == 0, out
    code, out = bounded("_one", 120)
    assert code == 0 and '"resumed": true' in out.lower(), out
    start = time.monotonic()
    code, _ = bounded("_hang", 2)
    assert code == 124 and time.monotonic() - start < 10
    value = {
        "status": "pass",
        "one_task": "pass",
        "failure_injection": "pass",
        "recovery": "pass",
        "resume": "pass",
        "timeout": "pass",
        "evaluation_label_invariance": "pass",
        "GPU": False,
        "cpu_threads": 1,
    }
    atomic(OUTPUT / "resource_audit.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("audit", "decision", "external", "all", "_one", "_hang"))
    parser.add_argument("--fault", action="store_true")
    args = parser.parse_args()
    if args.mode == "_hang":
        time.sleep(30)
        return
    if args.mode == "_one":
        result, resumed = external_one(0, args.fault)
        print(json.dumps({"status": "pass", "resumed": resumed, "methods": sorted(result)}))
        return
    if args.mode == "audit":
        print(json.dumps(audit(), sort_keys=True))
        return
    assert (OUTPUT / "resource_audit.json").exists(), "Run the resource audit first"
    assert json.loads((OUTPUT / "resource_audit.json").read_text())["status"] == "pass"
    if args.mode in ("decision", "all"):
        result = decision_analysis()
        atomic(OUTPUT / "decision_consequences.json", result)
        print(json.dumps({"decision": result["choice_change_counts"]}, sort_keys=True), flush=True)
    if args.mode in ("external", "all"):
        result = external_run()
        print(json.dumps({"external": result["summaries"], "changed": result["changed_vs_historical_frozen_selection"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
