"""
Failure autopsy.

    python inspect_failures.py data/cz_rail_20260714 instances.jsonl headroom.jsonl

An instance the generator PROVED solvable, on which a frontier model went 0/k,
is not evidence of difficulty. It is one of three things, and they need
different fixes:

  budget      n_calls pinned at MAX_CALLS, never submitted.
              The harness is failing, not the policy. Every "hard" instance is
              then an artifact of a constant I picked.

  semantics   submitted confidently and violated a constraint (avoid,
              max_transfers). The prompt or the tool output is unclear.

  search      submitted an infeasible itinerary -- tight transfer, wrong
              direction, legs that do not chain. Genuinely hard. Keep it.

Prints the reference the generator found, so you can see what the model missed.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from env import MAX_CALLS
from replay import World, hhmm

snap, inst_path, runs_path = (Path(a) for a in sys.argv[1:4])
w = World(snap)

instances = {json.loads(l)["id"]: json.loads(l) for l in inst_path.open()}
runs = [json.loads(l) for l in runs_path.open()]
df = pd.DataFrame(runs)

k = df.groupby("instance_id").size().max()
per = df.groupby("instance_id").solved.sum()


def classify(rollouts: list[dict]) -> str:
    """One label per instance, from its k rollouts."""
    if all(r["n_calls"] >= MAX_CALLS and not r["submitted"] for r in rollouts):
        return "budget"
    if all(r["feasible"] for r in rollouts):
        return "semantics"
    if any(not r["submitted"] for r in rollouts):
        return "mixed (some timed out)"
    return "search"


print(f"{'=' * 70}\nper-instance solve count")
hist = per.value_counts().reindex(range(k + 1), fill_value=0).sort_index()
for s, c in hist.items():
    tag = "  <- no gradient" if s in (0, k) else ""
    print(f"  {s}/{k}  {c:>3}{tag}")
print(f"\ninformative: {per.between(1, k - 1).mean():.0%}")

dead = sorted(per[per == 0].index)
print(f"\n{'=' * 70}\n{len(dead)} instances at 0/{k}. Diagnosing.\n")

labels = Counter()
for iid in dead:
    inst = instances[iid]
    rolls = [r for r in runs if r["instance_id"] == iid]
    label = classify(rolls)
    labels[label] += 1

    ref = inst["reference"]
    print(f"--- instance {iid}   [{label}]")
    print(f"  {inst['request']}")
    print(
        f"  reference: {hhmm(ref['departure'])} -> {hhmm(ref['arrival'])}, "
        f"{ref['transfers']} transfers"
    )
    for leg in ref["legs"]:
        print(f"      {leg['trip_id']:>8}  {w.name(leg['board'])} -> {w.name(leg['alight'])}")
    for r in rolls:
        first = (r["reasons"] or [""])[0][:70]
        print(f"      calls={r['n_calls']:>2} malformed={r['malformed']} {r['why']}")
        if first:
            print(f"          {first}")
    print()

print(f"{'=' * 70}\ndiagnosis of the 0/{k} instances:")
for label, c in labels.most_common():
    print(f"  {c:>3}  {label}")

budget_bound = (df.n_calls >= MAX_CALLS).mean()
print(f"\nrollouts pinned at MAX_CALLS={MAX_CALLS}: {budget_bound:.0%}")
if budget_bound > 0.15:
    print("\nTHE BUDGET IS BINDING. The histogram measures MAX_CALLS, not the task.")
    print("Raise it, or return more per departures() call, and re-run.")

# A defect worth fixing before training, whatever the diagnosis:
print(f"\n{'=' * 70}")
print("NOTE: never-submitting and submitting-something-infeasible score the")
print("same (-budget). A policy is therefore INDIFFERENT between guessing and")
print("continuing to search, and GRPO will teach it to guess early -- guessing")
print("is cheaper in tool calls. Consider a small extra penalty for a wrong")
print("submission, or a small reward for a well-formed one.")