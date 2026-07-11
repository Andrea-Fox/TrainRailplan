"""
Leak check on instances.jsonl.

    python check_leak.py data/cz_rail_20260714 instances.jsonl

A slack constraint exists so that "a constraint is present" carries no
information about whether it binds. If a trivial feature -- distance from the
avoid target to the origin-destination corridor -- separates binding from slack,
the policy can classify without searching, and the slack constraints are doing
nothing.

Prints the separation. Overlapping distributions are what we want.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

snap = Path(sys.argv[1])
inst = Path(sys.argv[2])

stations = pd.read_parquet(snap / "stations.parquet").set_index("station_id")
rows = [json.loads(l) for l in inst.open()]


def km(a, b):
    """Rough great-circle, adequate at this scale."""
    lat = np.radians((a[0] + b[0]) / 2)
    return 111.0 * np.hypot(a[0] - b[0], (a[1] - b[1]) * np.cos(lat))


def corridor_distance(avoid_id, o, d):
    """Distance from the avoid target to the straight line o->d, in km."""
    for s in (avoid_id, o, d):
        if s not in stations.index or not stations.loc[s, "geolocatable"]:
            return None
    p = stations.loc[avoid_id, ["stop_lat", "stop_lon"]].to_numpy(float)
    a = stations.loc[o, ["stop_lat", "stop_lon"]].to_numpy(float)
    b = stations.loc[d, ["stop_lat", "stop_lon"]].to_numpy(float)
    ab = b - a
    if not ab.any():
        return km(p, a)
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
    return km(p, a + t * ab)


binding, slack = [], []
for r in rows:
    av = r["constraints"]["avoid"]
    if not av:
        continue
    dist = corridor_distance(av[0], r["origin"], r["destination"])
    if dist is None:
        continue
    (binding if r["binds"].get("avoid") else slack).append(dist)

if not binding or not slack:
    print(f"binding={len(binding)} slack={len(slack)} -- need both to compare")
    sys.exit()

b, s = np.array(binding), np.array(slack)
print(f"avoid targets: {len(b)} binding, {len(s)} slack\n")
print("distance to the o->d corridor (km)")
for name, x in (("binding", b), ("slack", s)):
    print(f"  {name:>8}: median {np.median(x):6.1f}   p90 {np.quantile(x, .9):6.1f}   max {x.max():6.1f}")

# A single-threshold classifier is the crudest thing a policy could learn.
# If it separates well, the leak is open.
thresholds = np.linspace(0, max(b.max(), s.max()), 200)
acc = [(((b <= t).sum() + (s > t).sum()) / (len(b) + len(s)), t) for t in thresholds]
best, t = max(acc)
base = max(len(b), len(s)) / (len(b) + len(s))
print(f"\nbest single-threshold accuracy: {best:.0%} at {t:.0f} km")
print(f"majority-class baseline:        {base:.0%}")
if best - base > 0.20:
    print("\nLEAK: geometry alone predicts binding. Slack targets are too far away.")
else:
    print("\nok: geometry does not separate binding from slack.")