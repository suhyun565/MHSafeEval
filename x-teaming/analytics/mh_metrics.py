"""
analytics/mh_metrics.py
Parse X-Teaming all_results.json and show MHSafeEval-specific metrics:
  - ASR by category × role (trial-level: 1 patient × 1 combo = 1 trial)
  - ASR by category
  - ASR by role
  - Average turns to first jailbreak

Usage:
    python analytics/mh_metrics.py <TIMESTAMP>
    python analytics/mh_metrics.py <TIMESTAMP> -v      # verbose: per-combo detail
    python analytics/mh_metrics.py attacks/2025-.../all_results.json  # direct path
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def _parse_behavior_id(bid: str):
    """'{Category}__{Role}__pN' → (category, role, patient_id)"""
    parts = bid.split("__")
    cat  = parts[0].replace("_", " ").replace("-", "/") if len(parts) > 0 else "Unknown"
    role = parts[1].replace("_", " ").replace("-", "/") if len(parts) > 1 else "Unknown"
    pid  = parts[2].lstrip("p")                          if len(parts) > 2 else "?"
    return cat, role, pid


def load_results(path_or_timestamp: str) -> dict:
    p = Path(path_or_timestamp)

    # If it's a direct path to all_results.json
    if p.suffix == ".json" and p.exists():
        with open(p) as f:
            return json.load(f)

    # If it's a timestamp — look under attacks/
    attacks_dir = Path("attacks") / path_or_timestamp
    candidates  = [
        attacks_dir / "all_results.json",
        p / "all_results.json",
    ]
    for c in candidates:
        if c.exists():
            with open(c) as f:
                return json.load(f)

    sys.exit(f"❌ Cannot find all_results.json for: {path_or_timestamp}")


def compute_mh_metrics(results: dict) -> dict:
    """
    Parse X-Teaming all_results.json → MHSafeEval trial-level metrics.
    One trial = one behavior = one (category, role, patient) combo.
    """
    # behavior_number → {category, role, patient_id, jailbreak_achieved, jailbreak_turn}
    trials = []

    behaviors = results.get("behaviors", {})

    for bnum, bdata in behaviors.items():
        bid = (
            bdata.get("behavior", {}).get("BehaviorID")
            or bdata.get("behavior_details", {}).get("BehaviorID", "")
        )
        if not bid:
            # Try to extract from nested behavior_details
            bdet = bdata.get("behavior", {})
            bid  = bdet.get("BehaviorID", "")

        cat, role, pid = _parse_behavior_id(bid)

        # Jailbreak achieved if any strategy succeeded
        strategies      = bdata.get("strategies", [])
        jailbreak       = any(s.get("jailbreak_achieved", False) for s in strategies)
        jailbreak_turn  = None
        if jailbreak:
            turns = [
                s["jailbreak_turn"]
                for s in strategies
                if s.get("jailbreak_achieved") and s.get("jailbreak_turn")
            ]
            jailbreak_turn = min(turns) if turns else None

        trials.append({
            "behavior_id":    bid,
            "category":       cat,
            "role":           role,
            "patient_id":     pid,
            "jailbreak":      jailbreak,
            "jailbreak_turn": jailbreak_turn,
            "n_strategies":   len(strategies),
        })

    total   = len(trials)
    success = sum(1 for t in trials if t["jailbreak"])

    # ── By (category, role) ───────────────────────────────────────────────────
    by_combo = defaultdict(lambda: {"total": 0, "success": 0, "turns": []})
    for t in trials:
        k = (t["category"], t["role"])
        by_combo[k]["total"]   += 1
        by_combo[k]["success"] += int(t["jailbreak"])
        if t["jailbreak_turn"]:
            by_combo[k]["turns"].append(t["jailbreak_turn"])

    # ── By category ───────────────────────────────────────────────────────────
    by_cat = defaultdict(lambda: {"total": 0, "success": 0})
    for t in trials:
        by_cat[t["category"]]["total"]   += 1
        by_cat[t["category"]]["success"] += int(t["jailbreak"])

    # ── By role ───────────────────────────────────────────────────────────────
    by_role = defaultdict(lambda: {"total": 0, "success": 0})
    for t in trials:
        by_role[t["role"]]["total"]   += 1
        by_role[t["role"]]["success"] += int(t["jailbreak"])

    return {
        "total":    total,
        "success":  success,
        "overall_asr": success / total if total else 0.0,
        "trials":   trials,
        "by_combo": dict(by_combo),
        "by_cat":   dict(by_cat),
        "by_role":  dict(by_role),
    }


def _bar(asr: float, width: int = 10) -> str:
    filled = round(asr * width)
    return "█" * filled + "░" * (width - filled)


def print_report(m: dict, verbose: bool = False):
    W = 60
    print("\n" + "═" * W)
    print("MHSafeEval Results  (X-Teaming backend)")
    print("═" * W)
    print(
        f"  Total trials   : {m['total']}\n"
        f"  Successful     : {m['success']}\n"
        f"  Overall ASR    : {m['overall_asr']:.1%}\n"
    )

    # ── By category ───────────────────────────────────────────────────────────
    print("  By Category:")
    for cat, s in sorted(m["by_cat"].items()):
        asr = s["success"] / s["total"] if s["total"] else 0
        print(f"    {cat:<32} [{_bar(asr)}] {asr:.1%}  ({s['success']}/{s['total']})")

    print()

    # ── By role ───────────────────────────────────────────────────────────────
    print("  By Role:")
    for role in ["Enabler", "Facilitator", "Instigator", "Perpetrator"]:
        s   = m["by_role"].get(role, {"total": 0, "success": 0})
        asr = s["success"] / s["total"] if s["total"] else 0
        print(f"    {role:<15} [{_bar(asr)}] {asr:.1%}  ({s['success']}/{s['total']})")

    print()

    # ── Avg turns to jailbreak ────────────────────────────────────────────────
    all_turns = [t["jailbreak_turn"] for t in m["trials"] if t["jailbreak_turn"]]
    if all_turns:
        print(f"  Avg turns to jailbreak : {sum(all_turns)/len(all_turns):.1f}")
        print(f"  Min turns              : {min(all_turns)}")
        print(f"  Max turns              : {max(all_turns)}")
    print()

    # ── Verbose: per (category, role) ─────────────────────────────────────────
    if verbose:
        print("  Per Combo (Category × Role):")
        for (cat, role), s in sorted(m["by_combo"].items()):
            asr     = s["success"] / s["total"] if s["total"] else 0
            avg_t   = f"{sum(s['turns'])/len(s['turns']):.1f}" if s["turns"] else "—"
            print(
                f"    {cat:<30} × {role:<15} "
                f"[{_bar(asr)}] {asr:.1%}  ({s['success']}/{s['total']})  "
                f"avg_turn={avg_t}"
            )
        print()

    print("═" * W)


def main():
    parser = argparse.ArgumentParser(
        description="MHSafeEval metrics from X-Teaming results"
    )
    parser.add_argument("path",   help="Timestamp or path to all_results.json")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--save-json", default=None,
                        help="Save metrics to this JSON path")
    args = parser.parse_args()

    results = load_results(args.path)
    m       = compute_mh_metrics(results)
    print_report(m, verbose=args.verbose)

    if args.save_json:
        out = {
            "total":       m["total"],
            "success":     m["success"],
            "overall_asr": m["overall_asr"],
            "by_category": {
                cat: {
                    "n_trials":    s["total"],
                    "n_success":   s["success"],
                    "trial_asr":   s["success"] / s["total"] if s["total"] else 0.0,
                }
                for cat, s in m["by_cat"].items()
            },
            "by_role": {
                role: {
                    "n_trials":    s["total"],
                    "n_success":   s["success"],
                    "trial_asr":   s["success"] / s["total"] if s["total"] else 0.0,
                }
                for role, s in m["by_role"].items()
            },
            "by_combo": {
                f"{cat}-{role}": {
                    "n_trials":    s["total"],
                    "n_success":   s["success"],
                    "trial_asr":   s["success"] / s["total"] if s["total"] else 0.0,
                    "avg_turn_to_jailbreak": (
                        round(sum(s["turns"]) / len(s["turns"]), 2)
                        if s["turns"] else None
                    ),
                }
                for (cat, role), s in m["by_combo"].items()
            },
        }
        with open(args.save_json, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved → {args.save_json}")


if __name__ == "__main__":
    main()