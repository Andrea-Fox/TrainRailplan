"""
Search-quality autopsy.

    python compare_search.py headroom.jsonl

The budget-exhausted trace on an avoid-instance thrashed: repeated identical
calls, board==alight nonsense, no convergence. The question is whether that is
specific to binding constraints, or whether the search is bad everywhere and
the easy instances only got solved because they were short.

For every rollout, measure the search rather than eyeball it:
  calls          total tool calls
  distinct       unique (tool,args) calls
  redundancy     1 - distinct/calls   (how much was repeated)
  errors         calls the env rejected
  degenerate     board==alight calls (never sensible)

Then split by outcome. If solved traces are clean and only the budget-exhausted
ones thrash, the difficulty is real and specific. If solved traces are ALSO
redundant and just happened to be short, the interface invites thrashing and
that is the thing to fix.
"""

import json
import sys
from collections import Counter

import pandas as pd

runs = [json.loads(l) for l in open(sys.argv[1])]


def profile(r: dict) -> dict:
    calls = [s for s in r["trace"] if s["tool"] != "submit"]
    keys = [json.dumps([s["tool"], s["args"]], sort_keys=True, ensure_ascii=False) for s in calls]
    errors = sum(1 for s in calls if isinstance(s.get("result"), dict) and "error" in s["result"])
    degenerate = sum(
        1
        for s in calls
        if s["tool"] == "leg" and s["args"].get("board") == s["args"].get("alight")
    )
    n = len(calls)
    return {
        "why": r["why"],
        "any_binds": r["any_binds"],
        "calls": n,
        "distinct": len(set(keys)),
        "redundancy": round(1 - len(set(keys)) / n, 2) if n else 0.0,
        "errors": errors,
        "degenerate": degenerate,
    }


df = pd.DataFrame(profile(r) for r in runs if r["trace"])
df = df[df.calls > 0]

grp = "solved" if False else df.why.str.startswith("solved")
df["outcome"] = df.why.where(~df.why.str.startswith("never submitted"), "budget exhausted")
df["outcome"] = df["outcome"].where(df.why != "solved", "solved")

cols = ["calls", "distinct", "redundancy", "errors", "degenerate"]
print("by outcome:")
print(df.groupby("outcome")[cols].mean().round(2).to_string())
print("\ncounts:")
print(df.outcome.value_counts().to_string())

print("\nsolved vs budget-exhausted, redundancy distribution:")
for oc in ["solved", "budget exhausted"]:
    sub = df[df.outcome == oc].redundancy
    if len(sub):
        print(f"  {oc:<18} median {sub.median():.2f}  max {sub.max():.2f}  n={len(sub)}")

print("\ndegenerate (board==alight) calls total:", int(df.degenerate.sum()))
print("errors ignored then repeated -- mean errors per rollout:", round(df.errors.mean(), 1))

# The decisive contrast.
solved = df[df.outcome == "solved"]
if len(solved):
    print(f"\nsolved traces: mean {solved.calls.mean():.0f} calls, "
          f"{solved.redundancy.mean():.0%} redundant")
    if solved.redundancy.mean() > 0.2:
        print("  -> even SOLVED traces repeat themselves. The interface invites")
        print("     thrashing; the easy instances were just short enough to win")
        print("     anyway. Fix the tools, not only the difficulty.")
    else:
        print("  -> solved traces are clean. Thrashing is specific to the hard")
        print("     instances, and it is real difficulty, not an interface flaw.")