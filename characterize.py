"""
Per-instance characterization. No inference -- just line up every property of
each instance against how Claude did on it, and look.

    python characterize.py data/cz_rail_20260714 instances.jsonl headroom.jsonl
"""

import json
import sys
from pathlib import Path

import pandas as pd

from replay import World, hhmm

snap, inst_path, runs_path = (Path(a) for a in sys.argv[1:4])
w = World(snap)

instances = {json.loads(l)["id"]: json.loads(l) for l in inst_path.open()}
runs = [json.loads(l) for l in runs_path.open()]

by_inst: dict[int, list] = {}
for r in runs:
    by_inst.setdefault(r["instance_id"], []).append(r)

rows = []
for iid, rolls in sorted(by_inst.items()):
    inst = instances[iid]
    c = inst["constraints"]
    ref = inst["reference"]
    free = inst["free"]

    solved = sum(bool(x["solved"]) for x in rolls)
    whys = [x["why"] for x in rolls]

    # deadline slack: how much room between the reference arrival and the deadline
    slack_min = (c["arrive_before"] - ref["arrival"]) // 60

    # how much the constraints cost vs the unconstrained deadline solution
    dep_cost_min = (free["departure"] - ref["departure"]) // 60

    rows.append(
        {
            "id": iid,
            "solved": f"{solved}/{len(rolls)}",
            "mode": inst["mode"],
            "ref_transfers": ref["transfers"],
            "ref_dur_min": (ref["arrival"] - ref["departure"]) // 60,
            "slack_min": slack_min,
            "avoid": len(c["avoid"]),
            "max_tf": c["max_transfers"],
            "dep_after": c["depart_after"] > 0,
            "binds": inst["any_binds"],
            "orig_comp": w.component(inst["origin"]),
            "dest_comp": w.component(inst["destination"]),
            "origin": inst["origin_name"][:20],
            "destination": inst["destination_name"][:20],
        }
    )

df = pd.DataFrame(rows)
pd.set_option("display.width", 200, "display.max_columns", 30)
print(df.to_string(index=False))

print("\n\nper-instance, the request and what happened:")
for iid, rolls in sorted(by_inst.items()):
    inst = instances[iid]
    solved = sum(bool(x["solved"]) for x in rolls)
    print(f"\n[{iid}] {solved}/{len(rolls)}  {inst['request']}")
    ref = inst["reference"]
    print(f"     reference: {hhmm(ref['departure'])}->{hhmm(ref['arrival'])}, "
          f"{ref['transfers']} transfers, dur {(ref['arrival']-ref['departure'])//60}min")
    for leg in ref["legs"]:
        print(f"        {leg['trip_id']:>8}  {w.name(leg['board'])[:22]:<22} -> {w.name(leg['alight'])[:22]}")
    from collections import Counter
    for why, n in Counter(x["why"] for x in rolls).most_common():
        print(f"     {n}x  {why}")