"""Verify the published decision-consequence and corrected Davis records."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
decision = json.loads((ROOT / "expected" / "decision_consequences.json").read_text(encoding="utf-8"))
davis = json.loads((ROOT / "expected" / "davis_corrected_reselection.json").read_text(encoding="utf-8"))
audit = json.loads((ROOT / "expected" / "resource_audit.json").read_text(encoding="utf-8"))
protocol = json.loads((ROOT / "protocol_lock.json").read_text(encoding="utf-8"))

assert audit["status"] == "pass"
assert all(audit[key] == "pass" for key in (
    "one_task", "failure_injection", "recovery", "resume", "timeout", "evaluation_label_invariance"
))
assert decision["task_count"] == 21
assert decision["choice_change_counts"] == {
    "different_pair": 13, "orientation_reversal": 1, "unchanged": 7
}
assert decision["pareto_counts"] == {
    "A_dominates_B": 5, "B_dominates_A": 6, "same_outcome": 7, "tradeoff": 3
}
assert abs(decision["all_tasks"]["B_minus_A_regret"]["mean"] - 0.1438434198787846) < 1e-12
assert davis["donor_tasks"] == 21 and davis["external_tasks"] == 9
assert davis["selection_reads_evaluation_labels"] is False
assert davis["changed_vs_historical_frozen_selection"]["HIST-SAFE"] == 7
assert davis["summaries"]["HIST-SAFE"]["plugin_dual_success"] == 0
assert davis["summaries"]["HIST-SAFE"]["unresolved"] == 8
assert davis["summaries"]["Vina-MO"]["plugin_dual_success"] == 1
assert davis["summaries"]["Vina-MO"]["certified_positive"] == 1
assert protocol["learning_intervention"]["floor"] == decision["q25_floor"]

print(json.dumps({
    "status": "pass",
    "historical_tasks": decision["task_count"],
    "changed_choices": 14,
    "davis_tasks": davis["external_tasks"],
    "davis_hist_safe_unresolved": davis["summaries"]["HIST-SAFE"]["unresolved"],
}, sort_keys=True))
