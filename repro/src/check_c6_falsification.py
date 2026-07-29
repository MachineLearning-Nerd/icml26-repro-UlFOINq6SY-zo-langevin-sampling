"""Deterministic checker for the Claim 6 falsification campaign.

Reads outputs/c6_falsification.json (raw machine-readable evidence written by
verify_zo.py -> c6_falsify.run_campaign) and RECOMPUTES the decision from the
raw per-config numbers -- it does not trust the stored decision flag.

Exit code contract (release gate):
  0  -> a valid falsification IS present (controls pass AND every faithful
        completion violates "all 4 pb=10 configs reach mean FI < 0.01")
  1  -> falsification ABSENT (claim holds in >=1 faithful completion, or a
        control failed, so no assumption-valid counterexample exists)
  2  -> evidence file missing/malformed

Usage: uv run python repro/src/check_c6_falsification.py [path/to/json]
"""
from __future__ import annotations
import json
import os
import sys

THRESH = 0.01
FAITHFUL = ["primary_const_bp5", "variant_decay_bp5",
            "variant_const_bp2", "variant_const_sep8"]


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        root, "outputs", "c6_falsification.json")
    try:
        ev = json.load(open(path))
    except Exception as e:
        print(f"CHECK: cannot read evidence file {path}: {e}")
        return 2

    try:
        floor = ev["control_estimator_floor"]["FI_mean"]
        pos = ev["control_apmc_clean"]["FI_mean"]
        neg = ev["control_negative_pb3"]["FI_mean"]
        controls_ok = (floor < THRESH and pos < THRESH and neg > 0.05)
        print(f"CHECK: estimator floor mean FI = {floor:.5f}  "
              f"({'ok' if floor < THRESH else 'FAIL: floor >= 0.01'})")
        print(f"CHECK: positive control (gradient APMC, eps=0) = {pos:.5f}  "
              f"({'ok' if pos < THRESH else 'FAIL: cannot reach threshold'})")
        print(f"CHECK: negative control (pb=3) = {neg:.5f}  "
              f"({'ok' if neg > 0.05 else 'FAIL: not detected as bad'})")

        completions_fail = {}
        for key in FAITHFUL:
            cfgs = ev[key]["configs"]
            means = {c: v["FI_mean"] for c, v in cfgs.items()}
            claim_holds = all(m < THRESH for m in means.values())
            completions_fail[key] = not claim_holds
            worst = max(means, key=means.get)
            print(f"CHECK: {key}: claim "
                  f"{'HOLDS (all 4 < 0.01)' if claim_holds else 'violated'}; "
                  f"worst config {worst} mean FI={means[worst]:.4f}")
    except KeyError as e:
        print(f"CHECK: evidence file missing key: {e}")
        return 2

    falsified = controls_ok and all(completions_fail.values())
    print(f"CHECK: controls_ok={controls_ok}, "
          f"all_faithful_completions_fail={all(completions_fail.values())}")
    if falsified:
        print("CHECK RESULT: FALSIFICATION PRESENT (exit 0)")
        return 0
    print("CHECK RESULT: falsification ABSENT -> evidence status BLOCKED "
          "(exit 1); Claim 6 keeps its current verdict")
    return 1


if __name__ == "__main__":
    sys.exit(main())
