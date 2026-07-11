import random, pandas as pd
from replay import World
from solve import Constraints, SolverWorld, solve

snap = "data/cz_rail_20260714"
w, sw = World(snap), SolverWorld(World(snap), snap)
core = pd.read_parquet(f"{snap}/stations.parquet").query("component == 0").station_id.tolist()

random.seed(0)
for cap in (5, 7, 9, 12):
    random.seed(0)
    ts, gains = [], []
    for _ in range(100):
        o, d = random.sample(core, 2)
        a = solve(Constraints(o, d, depart_after=6*3600), sw, max_rounds=5)
        b = solve(Constraints(o, d, depart_after=6*3600), sw, max_rounds=cap)
        if b: ts.append(b.transfers)
        if a and b: gains.append((a.arrival - b.arrival) // 60)
    hi = sum(t >= cap - 1 for t in ts)
    print(f"cap={cap:>2}  at-boundary {hi:>3}  median gain vs cap5: {sorted(gains)[len(gains)//2]}min  max gain {max(gains)}min")