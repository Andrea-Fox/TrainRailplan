"""
Randomized cross-check on the real snapshot.

    python smoke_real.py data/cz_rail_20260714 [n]

Solves random origin/destination pairs and re-verifies every answer with
replay(). The solver reasons forward from the timetable; the verifier reasons
backward from the proposed itinerary. Any disagreement is a bug in one of them.

Also reports the no-solution rate, which predicts how many sampled instances
the generator will have to throw away.
"""

import random
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from replay import World, hhmm, replay
from solve import Constraints, SolverWorld, solve

snap = Path(sys.argv[1] if len(sys.argv) > 1 else "data/cz_rail_20260714")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
DEPART_AFTER = 6 * 3600

t0 = time.perf_counter()
w = World(snap)
sw = SolverWorld(w, snap)
print(f"indices built in {time.perf_counter() - t0:.1f}s")

stations = pd.read_parquet(snap / "stations.parquet")
core = stations[stations.component == 0].station_id.tolist()
print(f"{len(core)} stations in the largest component\n")

random.seed(0)
ok = rejected = unsolved = 0
transfers = Counter()
times = []

for i in range(N):
    o, d = random.sample(core, 2)
    t0 = time.perf_counter()
    sol = solve(Constraints(o, d, depart_after=DEPART_AFTER), sw)
    times.append(time.perf_counter() - t0)

    if sol is None:
        unsolved += 1
        continue

    r = replay(sol.legs, w)
    if not r.feasible:
        rejected += 1
        print(f"  REJECTED  {w.name(o)} -> {w.name(d)}")
        for reason in r.reasons:
            print(f"            {reason}")
        continue

    assert r.arrival == sol.arrival, f"arrival mismatch: {r.arrival} vs {sol.arrival}"
    assert r.transfers == sol.transfers
    ok += 1
    transfers[sol.transfers] += 1

times.sort()
print(f"verified   {ok}")
print(f"rejected   {rejected}   <- must be 0")
print(f"no solution {unsolved}  ({unsolved / N:.0%} of sampled pairs)")
print(f"\nsolve time  median {times[len(times)//2]*1000:.0f}ms  "
      f"p95 {times[int(len(times)*0.95)]*1000:.0f}ms  max {times[-1]*1000:.0f}ms")
print("\ntransfers used by the optimum:")
for k in sorted(transfers):
    print(f"  {k}: {transfers[k]:>4}  {'#' * (transfers[k] * 40 // max(transfers.values()))}")

if transfers and max(transfers) >= 4:
    print("\nWARNING: optimum uses 4 transfers; max_rounds=5 may be binding.")

# One worked example, printed in full.
random.seed(1)
for _ in range(50):
    o, d = random.sample(core, 2)
    sol = solve(Constraints(o, d, depart_after=DEPART_AFTER), sw)
    if sol and sol.transfers >= 2:
        print(f"\nexample: {w.name(o)} -> {w.name(d)}")
        print(f"  depart {hhmm(sol.departure)}  arrive {hhmm(sol.arrival)}  "
              f"{sol.transfers} transfers")
        for leg in sol.legs:
            print(f"    {leg.trip_id:>12}  {w.name(leg.board)} -> {w.name(leg.alight)}")
        break