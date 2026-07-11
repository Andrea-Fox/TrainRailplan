"""
Tests for solve(). Run: python test_solve.py

The load-bearing property: every itinerary solve() returns must independently
pass replay(). The solver reasons forward from the timetable; the verifier
reasons backward from a proposed answer. Agreement is evidence both are right.
"""

import shutil
import tempfile
from pathlib import Path

import pandas as pd

from replay import Leg, World, hhmm, replay
from solve import (
    HARD_CAP,
    Constraints,
    SolverWorld,
    satisfies,
    solve,
    solve_latest,
    violations,
)


def h(hh, mm=0):
    return hh * 3600 + mm * 60


#   T1: Praha 08:00 -> Kolin 09:00/09:02 -> Pardubice 09:30   (via Kolin)
#   T2: Kolin 09:02 -> HavlBrod 10:00                          (2 min: too tight)
#   T3: Kolin 09:20 -> HavlBrod 10:20                          (18 min: fine)
#   T6: Praha 08:10 -> HavlBrod 11:30                          (direct, slow)
#   T7: Praha 07:00 -> Kolin 07:45                             (early)
#
# A separate branch, built to discriminate the backward search. T8 calls at
# both X and Y, which are both one leg from D. Boarding at Z is possible only
# if the backward scan rides T8 as far as Y (the LATEST alighting), not X.
#   T8:  A 08:00 -> X 09:00 -> Z 09:30 -> Y 10:00
#   T9:  X 09:30 -> D 11:00
#   T10: Y 10:30 -> D 12:00
ROWS = [
    ("T1", 1, "Praha", h(8), h(8)),
    ("T1", 2, "Kolin", h(9), h(9, 2)),
    ("T1", 3, "Pardubice", h(9, 30), h(9, 30)),
    ("T2", 1, "Kolin", h(9, 2), h(9, 2)),
    ("T2", 2, "HavlBrod", h(10), h(10)),
    ("T3", 1, "Kolin", h(9, 20), h(9, 20)),
    ("T3", 2, "HavlBrod", h(10, 20), h(10, 20)),
    ("T6", 1, "Praha", h(8, 10), h(8, 10)),
    ("T6", 2, "HavlBrod", h(11, 30), h(11, 30)),
    ("T7", 1, "Praha", h(7), h(7)),
    ("T7", 2, "Kolin", h(7, 45), h(7, 45)),
    ("T8", 1, "A", h(8), h(8)),
    ("T8", 2, "X", h(9), h(9)),
    ("T8", 3, "Z", h(9, 30), h(9, 30)),
    ("T8", 4, "Y", h(10), h(10)),
    ("T9", 1, "X", h(9, 30), h(9, 30)),
    ("T9", 2, "D", h(11), h(11)),
    ("T10", 1, "Y", h(10, 30), h(10, 30)),
    ("T10", 2, "D", h(12), h(12)),
]
STATIONS = ["Praha", "Kolin", "Pardubice", "HavlBrod", "A", "X", "Y", "Z", "D"]

CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn

    return deco


def check(sol, c, w):
    """Every solution must pass the independent verifier."""
    assert sol is not None, "expected a solution"
    r = replay(sol.legs, w)
    assert r.feasible, f"solver emitted an itinerary replay() rejects: {r.reasons}"
    assert r.arrival == sol.arrival, f"{r.arrival} != {sol.arrival}"
    assert r.transfers == sol.transfers
    return r


@case("direct: earliest arrival")
def _(w, sw):
    c = Constraints("Praha", "Pardubice", depart_after=h(7, 30))
    sol = solve(c, sw)
    r = check(sol, c, w)
    assert sol.arrival == h(9, 30) and sol.transfers == 0


@case("prefers the transfer when it arrives earlier")
def _(w, sw):
    # T6 direct arrives 11:30. T7 (Praha 07:00 -> Kolin 07:45) then T2
    # (Kolin 09:02 -> HavlBrod 10:00) arrives 10:00. Earliest arrival is
    # indifferent to the 77-minute wait at Kolin -- correctly so.
    c = Constraints("Praha", "HavlBrod", depart_after=h(6))
    sol = solve(c, sw)
    check(sol, c, w)
    assert sol.arrival == h(10), sol.arrival
    assert sol.transfers == 1
    assert [l.trip_id for l in sol.legs] == ["T7", "T2"]


@case("max_transfers=0 forces the slow direct train")
def _(w, sw):
    c = Constraints("Praha", "HavlBrod", depart_after=h(6), max_transfers=0)
    sol = solve(c, sw)
    check(sol, c, w)
    assert sol.transfers == 0 and sol.arrival == h(11, 30)


@case("the 2-minute transfer is never used")
def _(w, sw):
    c = Constraints("Praha", "HavlBrod", depart_after=h(7, 50))
    sol = solve(c, sw)
    check(sol, c, w)
    # T1 arrives Kolin 09:00; T2 departs 09:02. Only 120s < 180s.
    assert "T2" not in [l.trip_id for l in sol.legs]


@case("avoid: cannot pass through a forbidden station")
def _(w, sw):
    c = Constraints("Praha", "Pardubice", depart_after=h(6), avoid=frozenset({"Kolin"}))
    assert solve(c, sw) is None, "T1 passes through Kolin; must be rejected"


@case("avoid: origin or destination forbidden is trivially infeasible")
def _(w, sw):
    c = Constraints("Praha", "Kolin", avoid=frozenset({"Kolin"}))
    assert solve(c, sw) is None


@case("arrive_before prunes but stays exact")
def _(w, sw):
    c = Constraints("Praha", "HavlBrod", depart_after=h(6), arrive_before=h(10, 30))
    sol = solve(c, sw)
    check(sol, c, w)
    assert sol.arrival <= h(10, 30)

    tight = Constraints("Praha", "HavlBrod", depart_after=h(6), arrive_before=h(9))
    assert solve(tight, sw) is None


@case("depart_after is respected")
def _(w, sw):
    c = Constraints("Praha", "Kolin", depart_after=h(7, 30))
    sol = solve(c, sw)
    r = check(sol, c, w)
    assert r.departure >= h(7, 30)


@case("unreachable pair returns None, not a crash")
def _(w, sw):
    c = Constraints("Pardubice", "Praha", depart_after=h(6))
    assert solve(c, sw) is None  # nothing runs westbound in this toy world


# --- constraint evaluation --------------------------------------------------


@case("violations: satisfied constraints score zero")
def _(w, sw):
    c = Constraints("Praha", "HavlBrod", depart_after=h(6), max_transfers=2)
    sol = solve(c, sw)
    r = replay(sol.legs, w)
    assert satisfies(r, c), violations(r, c)


@case("violations: measures magnitude, not just a flag")
def _(w, sw):
    # Solve unconstrained, then score against a constraint it violates.
    loose = Constraints("Praha", "HavlBrod", depart_after=h(6))
    sol = solve(loose, sw)
    r = replay(sol.legs, w)

    strict = Constraints("Praha", "HavlBrod", depart_after=h(6), arrive_before=h(9, 30))
    v = violations(r, strict)
    assert v["arrive_before"] == 0.5, v  # arrives 10:00, wanted 09:30
    assert not satisfies(r, strict)


@case("violations: avoid counts pass-throughs, not just transfers")
def _(w, sw):
    c = Constraints("Praha", "Pardubice", depart_after=h(6))
    sol = solve(c, sw)
    r = replay(sol.legs, w)
    # T1 never stops for a change at Kolin, but it does pass through.
    strict = Constraints(
        "Praha", "Pardubice", depart_after=h(6), avoid=frozenset({"Kolin"})
    )
    assert violations(r, strict)["avoid"] == 1


# --- the non-triviality filter ---------------------------------------------


@case("non-triviality: unconstrained optimum must violate something")
def _(w, sw):
    # max_transfers=1 is decorative: the unconstrained optimum uses 1 transfer.
    decorative = Constraints("Praha", "HavlBrod", depart_after=h(6), max_transfers=1)
    free = solve(decorative.unconstrained(), sw)
    r = replay(free.legs, w)
    assert satisfies(r, decorative), "expected this instance to be trivial"

    # max_transfers=0 bites: the unconstrained optimum uses 1 transfer.
    binding = Constraints("Praha", "HavlBrod", depart_after=h(6), max_transfers=0)
    assert not satisfies(r, binding), "expected this instance to be non-trivial"


@case("converged=True when the search reaches a fixpoint")
def _(w, sw):
    # The toy world is small enough that everything settles inside the cap.
    c = Constraints("Praha", "HavlBrod", depart_after=h(6))
    sol = solve(c, sw)
    assert sol.converged


@case("converged=False when the round cap stops the search early")
def _(w, sw):
    # One round: stations were still improving when the cap hit. The answer is
    # still exact for the bounded problem -- 'at most 0 transfers' -- but it is
    # not the global optimum, and converged says so.
    c = Constraints("Praha", "Kolin", depart_after=h(6))
    sol = solve(c, sw, max_rounds=1)
    assert sol is not None and not sol.converged


@case("a bounded answer is still exact for the bounded problem")
def _(w, sw):
    # max_rounds=1 forbids transfers. The best direct Praha->Kolin train
    # departing after 07:30 is T1 at 08:00. Not converged, but not wrong.
    sol = solve(Constraints("Praha", "Kolin", depart_after=h(7, 30)), sw, max_rounds=1)
    check(sol, Constraints("Praha", "Kolin", depart_after=h(7, 30)), w)
    assert sol.transfers == 0 and sol.departure == h(8)


@case("max_transfers is honoured, never clamped to max_rounds")
def _(w, sw):
    # A caller asking for 6 transfers must not be silently given a 2-round
    # search. Solvable-with-6 instances would come back None.
    c = Constraints("Praha", "HavlBrod", depart_after=h(6), max_transfers=6)
    sol = solve(c, sw, max_rounds=3)
    assert sol is not None, "max_rounds truncated the caller's max_transfers"


@case("absurd transfer budgets raise rather than truncate")
def _(w, sw):
    c = Constraints("Praha", "HavlBrod", max_transfers=HARD_CAP + 5)
    try:
        solve(c, sw)
    except AssertionError:
        return
    raise AssertionError("expected HARD_CAP to reject the request")


# --- latest departure --------------------------------------------------------


@case("latest: departs as late as possible for a given deadline")
def _(w, sw):
    # Praha -> Kolin: T7 at 07:00 (arr 07:45), T1 at 08:00 (arr 09:00).
    # Deadline 09:30 -> should take T1, not T7.
    c = Constraints("Praha", "Kolin", depart_after=h(6), arrive_before=h(9, 30))
    sol = solve_latest(c, sw)
    check(sol, c, w)
    assert sol.departure == h(8), hhmm(sol.departure)
    assert [l.trip_id for l in sol.legs] == ["T1"]


@case("latest: a tighter deadline forces an earlier departure")
def _(w, sw):
    c = Constraints("Praha", "Kolin", depart_after=h(6), arrive_before=h(8, 30))
    sol = solve_latest(c, sw)
    check(sol, c, w)
    assert sol.departure == h(7), hhmm(sol.departure)  # T1 arrives 09:00, too late


@case("latest: arrive_before actually binds")
def _(w, sw):
    loose = solve_latest(
        Constraints("Praha", "Kolin", depart_after=h(6), arrive_before=h(9, 30)), sw
    )
    tight = solve_latest(
        Constraints("Praha", "Kolin", depart_after=h(6), arrive_before=h(8, 30)), sw
    )
    assert tight.departure < loose.departure, "the deadline changed nothing"


@case("latest: infeasible deadline returns None")
def _(w, sw):
    c = Constraints("Praha", "Kolin", depart_after=h(6), arrive_before=h(6, 30))
    assert solve_latest(c, sw) is None


@case("latest: respects depart_after")
def _(w, sw):
    c = Constraints("Praha", "Kolin", depart_after=h(7, 30), arrive_before=h(9, 30))
    sol = solve_latest(c, sw)
    r = check(sol, c, w)
    assert r.departure >= h(7, 30)


@case("latest: respects avoid, including pass-throughs")
def _(w, sw):
    c = Constraints(
        "Praha",
        "Pardubice",
        depart_after=h(6),
        arrive_before=h(12),
        avoid=frozenset({"Kolin"}),
    )
    assert solve_latest(c, sw) is None


@case("latest: respects max_transfers")
def _(w, sw):
    c = Constraints(
        "Praha", "HavlBrod", depart_after=h(6), arrive_before=h(12), max_transfers=0
    )
    sol = solve_latest(c, sw)
    check(sol, c, w)
    assert sol.transfers == 0


@case("duality: feasible under latest iff earliest arrival beats the deadline")
def _(w, sw):
    for deadline in (h(8), h(9), h(10), h(10, 30), h(12)):
        c = Constraints("Praha", "HavlBrod", depart_after=h(6), arrive_before=deadline)
        early = solve(Constraints("Praha", "HavlBrod", depart_after=h(6)), sw)
        late = solve_latest(c, sw)
        expect = early is not None and early.arrival <= deadline
        assert (late is not None) == expect, f"deadline {hhmm(deadline)}"


@case("duality: latest never departs before earliest-arrival's departure")
def _(w, sw):
    c = Constraints("Praha", "HavlBrod", depart_after=h(6), arrive_before=h(12))
    early = solve(Constraints("Praha", "HavlBrod", depart_after=h(6)), sw)
    late = solve_latest(c, sw)
    assert late.departure >= early.departure


# --- cases that discriminate the backward search ----------------------------
# Each of these was added because a mutant survived without it.


@case("latest: rides a trip to its latest useful alighting, not its first")
def _(w, sw):
    # Boarding at Z is reachable only if the backward scan rides T8 past X to
    # Y. If it stops at the first marked alighting (X), Z is never labelled and
    # this instance looks infeasible.
    c = Constraints("Z", "D", depart_after=h(6), arrive_before=h(12))
    sol = solve_latest(c, sw)
    assert sol is not None, "backward scan took the earliest alighting, not the latest"
    check(sol, c, w)
    assert sol.departure == h(9, 30)
    assert [l.trip_id for l in sol.legs] == ["T8", "T10"]


@case("latest: min_transfer is enforced backwards too")
def _(w, sw):
    # The only way to reach HavlBrod by 10:10 departing after 07:50 is
    # T1 -> T2, a 2-minute change. Must be rejected.
    c = Constraints("Praha", "HavlBrod", depart_after=h(7, 50), arrive_before=h(10, 10))
    assert solve_latest(c, sw) is None, "backward scan ignored the minimum transfer"


@case("latest: depart_after can make an instance infeasible")
def _(w, sw):
    # Only T7 (07:00) reaches Kolin by 08:00, and it departs too early.
    # Maximizing departure alone would never surface this -- the bound must be
    # enforced during the scan.
    c = Constraints("Praha", "Kolin", depart_after=h(7, 30), arrive_before=h(8))
    assert solve_latest(c, sw) is None, "backward scan ignored depart_after"


def main():
    tmp = Path(tempfile.mkdtemp())
    try:
        pd.DataFrame(
            ROWS, columns=["trip_id", "stop_sequence", "station_id", "arr", "dep"]
        ).to_parquet(tmp / "stop_times.parquet", index=False)
        n = len(STATIONS)
        pd.DataFrame(
            {
                "station_id": STATIONS,
                "stop_name": STATIONS,
                "stop_lat": [50.0] * n,
                "stop_lon": [14.0] * n,
                "geolocatable": [True] * n,
                "n_stop_ids": [1] * n,
                "component": [0] * n,
            }
        ).to_parquet(tmp / "stations.parquet", index=False)

        w = World(tmp)
        sw = SolverWorld(w, tmp)

        failed = 0
        for name, fn in CASES:
            try:
                fn(w, sw)
                print(f"  pass  {name}")
            except AssertionError as e:
                failed += 1
                print(f"  FAIL  {name}\n        {e}")
        print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
        return 1 if failed else 0
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    raise SystemExit(main())