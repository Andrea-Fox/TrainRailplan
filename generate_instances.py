"""
Instance generator.

    python generate_instances.py data/cz_rail_20260714 --n 2000 --out instances.jsonl

Objective: latest departure subject to a deadline. "I need to be in Brno by
18:00 -- when do I leave?" Under earliest arrival, `arrive_before` could never
bind (the free arrival IS the minimum), so a tighter deadline was infeasible
rather than tight, and the rejection filter quietly selected against it.

THE SELECTION-BIAS TRAP, and the defence:

  Sampling a constraint value and discarding the instance when it turns out
  infeasible biases the surviving dataset toward loose constraints -- silently,
  and in exactly the direction that makes the task easier. So constraints are
  *constructed* feasible: propose the tightest value, relax one step at a time
  until the solver returns something. Rejections are rare and are logged.

THE REWARD-HACKING TRAP, and the defence:

  If every constraint always binds, and always sits a fixed offset from the
  free solution, a policy can ignore the request, compute the greedy answer,
  and shave a known amount. So:
    1. Offsets are randomized over a wide range.
    2. Which constraints appear is randomized; absence is informative.
    3. A fraction are deliberately slack. If "present" implied "binding",
       presence alone would be exploitable without reading the value.

Nothing here is shown to the policy except `request`.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from replay import World, hhmm, replay
from solve import (
    DEFAULT_MAX_ROUNDS,
    Constraints,
    SolverWorld,
    satisfies,
    solve,
    solve_latest,
    violations,
)

HOUR = 3600
LONG_TRIP_SEC = 90 * 60  # a trip this long serves places worth travelling to

MODES = ["hub_hub", "hub_leaf", "leaf_leaf"]
MODE_P = [0.25, 0.55, 0.20]

# Target distribution over transfer counts in the KEPT set. The generator
# rejects an instance whose depth bucket is already at its quota, so the output
# converges to this shape by construction rather than by post-hoc subsampling.
#
# Weighted shallow on purpose: Qwen cannot yet solve a direct train, so SFT must
# teach the easy cases heavily before the hard ones. 4+ transfers are excluded
# entirely -- even a frontier model fails those, so they are not a training
# target. Set via --depth-dist to override.
DEPTH_DIST = {0: 0.15, 1: 0.35, 2: 0.35, 3: 0.15}


def station_weights(snapshot: Path, core: list[str]) -> np.ndarray:
    """Hub prior. This feed has no service-class field -- is_regional and
    is_night are constant zero -- so trip duration is the proxy: a station
    served by long-running trips is somewhere people travel to."""
    st = pd.read_parquet(snapshot / "stop_times.parquet")
    span = st.groupby("trip_id").agg(lo=("dep", "min"), hi=("arr", "max"))
    long_trips = set(span[(span.hi - span.lo) >= LONG_TRIP_SEC].index)

    w = st[st.trip_id.isin(long_trips)].groupby("station_id").trip_id.nunique()
    w = w.reindex(core).fillna(0.0)
    if w.sum() == 0:  # pathological feed; degrade gracefully
        w = st.groupby("station_id").trip_id.nunique().reindex(core).fillna(0.0)
    return (w / w.sum()).to_numpy()


def sample_pair(rng, np_rng, mode: str, core, weights) -> tuple[str, str]:
    def hub():
        return core[int(np_rng.choice(len(core), p=weights))]

    def leaf():
        return rng.choice(core)

    if mode == "hub_hub":
        return hub(), hub()
    if mode == "hub_leaf":
        o, d = hub(), leaf()
        return (o, d) if rng.random() < 0.5 else (d, o)
    return leaf(), leaf()


def tighten_transfers(c: Constraints, free_transfers: int, sw) -> Constraints | None:
    """Tightest transfer budget that still admits a solution. Steps up from the
    binding value instead of sampling and discarding."""
    for k in range(max(0, free_transfers - 1), free_transfers):
        cc = replace(c, max_transfers=k)
        if solve_latest(cc, sw) is not None:
            return cc
    return None


def pick_avoid(rng, c: Constraints, path: list[str], sw, tries: int = 4):
    """An avoid target that blocks the free route without disconnecting the
    pair. Try a few before giving up."""
    candidates = [s for s in path if s not in (c.origin, c.destination)]
    rng.shuffle(candidates)
    for s in candidates[:tries]:
        cc = replace(c, avoid=frozenset({s}))
        if solve_latest(cc, sw) is not None:
            return cc
    return None


def plausible_slack_avoid(rng, path: list[str], coords: pd.DataFrame, core: set):
    """A slack avoid target that a traveller might plausibly name.

    Sampling uniformly over all stations LEAKS: a station hundreds of km off
    the corridor is obviously irrelevant, so the policy can classify a
    constraint as binding-or-not by geography alone, without searching. The
    slack constraints then stop doing the job they exist for.

    Instead, sample inside the bounding box of the free route, padded, and
    exclude the route itself. Near the corridor, off the chosen path.
    """
    on_path = set(path)
    pts = coords.reindex([s for s in path if s in coords.index])
    pts = pts[pts.geolocatable]
    if pts.empty:
        return None

    pad = 0.15  # degrees, ~17 km
    box = coords[
        coords.geolocatable
        & coords.stop_lat.between(pts.stop_lat.min() - pad, pts.stop_lat.max() + pad)
        & coords.stop_lon.between(pts.stop_lon.min() - pad, pts.stop_lon.max() + pad)
    ]
    cands = sorted(set(box.index) & core - on_path)
    return rng.choice(cands) if cands else None


def build(rng, c0: Constraints, free, coords, core: set, sw) -> tuple[Constraints, dict]:
    """Layer constraints onto the deadline query, preserving feasibility at
    every step. Returns the constraint set and which parts were made to bind."""
    c = c0
    binds = {"arrive_before": True}  # the objective's constraint; binds by definition

    if rng.random() < 0.65 and free.transfers > 0:
        if rng.random() < 0.8:
            # Feasibility is monotone in the budget: if free.transfers - 1 is
            # infeasible, no tighter value is either. One attempt is enough.
            tight = tighten_transfers(c, free.transfers, sw)
            if tight is not None:
                c, binds["max_transfers"] = tight, True
        else:
            c = replace(c, max_transfers=free.transfers + rng.randint(0, 2))
            binds["max_transfers"] = False

    if rng.random() < 0.55:
        if rng.random() < 0.7:
            blocked = pick_avoid(rng, c, free.stations_visited, sw)
            if blocked is not None:
                c, binds["avoid"] = blocked, True
        else:
            s = plausible_slack_avoid(rng, free.stations_visited, coords, core)
            if s is not None:
                c = replace(c, avoid=frozenset({s}))
                binds["avoid"] = False

    # depart_after is slack under this objective: maximizing departure already
    # avoids early trains, so lowering the bound cannot change the answer. It
    # only bites by making an instance infeasible. Present half the time, so
    # that "a bound appeared" carries no information; rounded to the quarter
    # hour, because "no earlier than 04:33" is not a thing anyone says.
    if rng.random() < 0.5:
        floor = free.departure - rng.randint(1, 6) * HOUR
        floor = max(4 * HOUR, (floor // 900) * 900)
        c = replace(c, depart_after=floor)
        binds["depart_after"] = False
    else:
        c = replace(c, depart_after=0)

    return c, binds


def render(c: Constraints, w: World) -> str:
    """Templated request. Deliberately NOT an LLM paraphrase yet: this keeps
    'can the agent search' separate from 'can the agent parse vague language'.
    Answer the first question before adding the second."""
    parts = [f"I need to get from {w.name(c.origin)} to {w.name(c.destination)}"]
    parts.append(f"arriving by {hhmm(c.arrive_before)}")
    if c.max_transfers is not None:
        n = c.max_transfers
        parts.append(
            "with no changes" if n == 0 else f"with at most {n} change{'s' if n > 1 else ''}"
        )
    if c.avoid:
        parts.append("avoiding " + ", ".join(sorted(w.name(s) for s in c.avoid)))
    if c.depart_after > 0:
        parts.append(f"leaving no earlier than {hhmm(c.depart_after)}")
    return ", ".join(parts) + "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="instances.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-duration-h", type=float, default=6.0)
    ap.add_argument("--slack-h", type=float, default=3.0, help="deadline headroom")
    ap.add_argument("--max-attempts", type=int, default=50)
    args = ap.parse_args()

    snap = Path(args.snapshot)
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    w = World(snap)
    sw = SolverWorld(w, snap)
    stations = pd.read_parquet(snap / "stations.parquet")
    core = stations[stations.component == 0].station_id.tolist()
    core_set = set(core)
    coords = stations.set_index("station_id")[["stop_lat", "stop_lon", "geolocatable"]]
    weights = station_weights(snap, core)

    kept, rejected = [], Counter()
    attempts, limit = 0, args.n * args.max_attempts

    # Per-bucket quota. An instance whose depth bucket is full is rejected, so
    # the kept set converges to DEPTH_DIST. Buckets not in the dist are excluded.
    depth_quota = {k: round(args.n * v) for k, v in DEPTH_DIST.items()}
    depth_kept = Counter()

    while len(kept) < args.n and attempts < limit:
        attempts += 1
        mode = rng.choices(MODES, MODE_P)[0]
        o, d = sample_pair(rng, np_rng, mode, core, weights)
        if o == d:
            rejected["same station"] += 1
            continue

        # Forward solve fixes the horizon and screens out cross-country slogs.
        # NOTE: do NOT filter on `converged`. That flag means the whole network
        # stopped improving, which at 5 rounds essentially never happens. The
        # answer is exact for the bounded problem regardless.
        earliest = rng.randint(5, 16) * HOUR
        fwd = solve(Constraints(o, d, depart_after=earliest), sw)
        if fwd is None:
            rejected["unreachable"] += 1
            continue

        # Cheap pre-gate: skip depths never targeted, and skip a bucket that is
        # full even before constraints (which can only raise depth, not lower
        # it much). The authoritative gate is on ref.transfers below.
        depth = fwd.transfers
        if depth not in depth_quota and depth > max(depth_quota):
            rejected[f"depth {depth} out of range"] += 1
            continue

        if fwd.duration > args.max_duration_h * HOUR:
            rejected["journey too long"] += 1
            continue

        # The deadline: reachable, with randomized headroom. This is the query.
        deadline = fwd.arrival + rng.randint(0, int(args.slack_h * 60)) * 60
        c0 = Constraints(o, d, depart_after=earliest, arrive_before=deadline)

        free = solve_latest(c0, sw)
        if free is None:
            rejected["deadline unreachable"] += 1
            continue
        free_report = replay(free.legs, w)
        assert free_report.feasible, "solver emitted an itinerary replay rejects"

        c, binds = build(rng, c0, free_report, coords, core_set, sw)
        ref = solve_latest(c, sw)
        if ref is None:
            rejected["constraints unsatisfiable"] += 1
            continue

        if ref.transfers >= DEFAULT_MAX_ROUNDS - 1 and c.max_transfers is None:
            # Reference sits at the transfer cap, so it may be far from optimal.
            # A bad reference is a bad SFT target and a bad baseline.
            rejected["reference at transfer cap"] += 1
            continue

        ref_report = replay(ref.legs, w)
        assert ref_report.feasible
        assert satisfies(ref_report, c), violations(ref_report, c)

        # The stored reference -- not the unconstrained fwd solve -- is what
        # simulate_traces keys on. A max_transfers constraint can change the
        # depth between fwd and ref, so gate the quota on ref.transfers, the
        # number that actually shapes the trace distribution.
        rdepth = ref.transfers
        if rdepth not in depth_quota or depth_kept[rdepth] >= depth_quota[rdepth]:
            rejected[f"depth {rdepth} bucket full (post-constraint)"] += 1
            continue
        depth_kept[rdepth] += 1

        kept.append(
            {
                "id": len(kept),
                "mode": mode,
                "objective": "latest_departure",
                "converged": bool(ref.converged),  # diagnostic, not a filter
                "request": render(c, w),
                "binds": {k: bool(v) for k, v in binds.items()},
                "any_binds": any(v for k, v in binds.items() if k != "arrive_before"),
                "origin": o,
                "destination": d,
                "origin_name": w.name(o),
                "destination_name": w.name(d),
                "constraints": {
                    "depart_after": c.depart_after,
                    "arrive_before": c.arrive_before,
                    "max_transfers": c.max_transfers,
                    "avoid": sorted(c.avoid),
                },
                "reference": {
                    "legs": [asdict(l) for l in ref.legs],
                    "departure": ref.departure,
                    "arrival": ref.arrival,
                    "transfers": ref.transfers,
                },
                "free": {  # deadline only, no preferences: the difficulty gap
                    "departure": free.departure,
                    "arrival": free.arrival,
                    "transfers": free.transfers,
                },
            }
        )

    out = Path(args.out)
    with out.open("w") as f:
        for inst in kept:
            f.write(json.dumps(inst, ensure_ascii=False) + "\n")

    print(f"kept {len(kept)} of {attempts} attempts ({len(kept)/max(attempts,1):.0%})\n")
    print("achieved depth distribution (target in parens):")
    for k in sorted(depth_quota):
        got = depth_kept[k]
        tgt = DEPTH_DIST[k]
        print(f"  {k} transfers: {got:>5}  ({got/max(len(kept),1):.0%}, target {tgt:.0%})")
    print("\nrejected:")
    for why, n in rejected.most_common():
        print(f"  {n:>6}  {why}")
    if not kept:
        return

    df = pd.DataFrame(kept)
    ref = pd.json_normalize(df.reference)
    free_df = pd.json_normalize(df.free)
    cons = pd.json_normalize(df.constraints)
    binds = pd.json_normalize(df.binds)

    print("\nby mode:")
    print(df.groupby("mode").size().to_string())

    print("\nconstraint present / of which binding:")
    for col in ["max_transfers", "avoid"]:
        present = (
            cons[col].notna() if col == "max_transfers" else cons.avoid.map(len).gt(0)
        )
        b = binds.get(col)
        frac = 0.0 if b is None else b[present].eq(True).mean()
        print(f"  {col:>14}: {present.mean():5.0%}   binding {frac:.0%}")
    print(f"  {'any binding':>14}: {df.any_binds.mean():5.0%}")

    print("\nreference transfers:")
    print(ref.transfers.value_counts().sort_index().to_string())

    lost = (free_df.departure - ref.departure) / 60
    print("\ncost of the preferences (free departure - ref departure):")
    print(f"  median {lost.median():.0f} min   p90 {lost.quantile(0.9):.0f} min")

    slack = (cons.arrive_before - ref.arrival) / 60
    print(f"\ndeadline slack in the reference: median {slack.median():.0f} min")

    print("\nexample requests:")
    for r in df.request.sample(min(3, len(df)), random_state=0):
        print(f"  {r}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()