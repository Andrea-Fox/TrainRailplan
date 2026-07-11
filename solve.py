"""
solve(): the reference solver. Earliest arrival, subject to a constraint set.

Three jobs, in order of importance:

  1. Oracle          -- does a feasible itinerary exist for this instance?
  2. Non-triviality  -- does the *unconstrained* optimum violate a constraint?
                        If not, the constraints are decorative and the policy
                        can ignore the user's request and still score.
  3. Baseline        -- the gap the trained policy is measured against.

It is NOT the deployment artifact. The policy never calls it. It sees only
local lookups (departures, leg, find_station) and must search for itself.

Algorithm: round-based label setting (RAPTOR without the route abstraction).
Round k settles the earliest arrival reachable using exactly k legs, so
`max_transfers = k - 1` falls out of the round index rather than needing a
state dimension. Trip scanning is fast enough at this scale (~9k trips).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from replay import DAY, Leg, World

INF = 1 << 40

# Rounds = legs, so max_transfers = rounds - 1.
#
# Default 5 (up to 4 transfers). Measured on the 2026-07-14 CZ rail snapshot,
# 100 random pairs in the largest component departing after 06:00: at 5 rounds
# 31% of optima sit on the boundary; at 7, two; at 9, none, and the arrival
# gain saturates (max 148 min vs cap 5, unchanged at 12).
#
# We keep 5 anyway. The instances the cap truncates are rural-to-rural journeys
# routed through Praha and Brno -- eight-hour slogs nobody requests. They are
# out of the target distribution, so the sampler must exclude them regardless.
#
# The cap DEFINES the problem: solve() returns the exact earliest arrival using
# at most `rounds - 1` transfers. That is always a correct answer to a
# well-posed question. `Solution.converged` reports the stronger property --
# the search hit a fixpoint, so the answer is also the unbounded optimum -- but
# on a 2,466-station network that needs far more than 5 rounds and is normally
# False. It is a diagnostic, not a filter.
DEFAULT_MAX_ROUNDS = 5

# Above this, refuse. A request needing 12 legs is a bug in the caller, not a
# journey. Better to raise than to quietly explore less than asked.
HARD_CAP = 12

# Above this, refuse. A request needing 12 legs is a bug in the caller, not a
# journey. Better to raise than to quietly explore less than asked.
HARD_CAP = 12


@dataclass(frozen=True)
class Constraints:
    """The user's request, as a machine-checkable tuple."""

    origin: str  # station_id
    destination: str  # station_id
    depart_after: int = 0  # seconds since service-day midnight
    arrive_before: int | None = None
    max_transfers: int | None = None
    avoid: frozenset[str] = frozenset()  # station_ids

    def unconstrained(self) -> "Constraints":
        """Same journey, no preferences. Used for the non-triviality check."""
        return Constraints(self.origin, self.destination, depart_after=self.depart_after)


@dataclass
class Solution:
    legs: list[Leg]
    arrival: int
    departure: int
    transfers: int

    # The solution is ALWAYS exact for the bounded problem: earliest arrival
    # (or latest departure) using at most `rounds - 1` transfers.
    #
    # `converged` is the stronger, rarer claim: the label-setting search reached
    # a fixpoint -- no station anywhere improved -- so the answer is also the
    # global optimum, unbounded transfers. On a 2,466-station network this
    # takes far more than 5 rounds, so at the default cap it is usually False.
    # Do not use it as a filter; it does not mean "bad instance".
    converged: bool = False

    @property
    def duration(self) -> int:
        return self.arrival - self.departure


class SolverWorld:
    """Indices the solver needs but replay() does not. Built once, reused."""

    def __init__(self, world: World, snapshot):
        st = pd.read_parquet(f"{snapshot}/stop_times.parquet")
        st = st.sort_values(["trip_id", "stop_sequence"])

        # trip_id -> ordered list of (seq, station_id, arr, dep)
        self.trip_calls: dict[str, list[tuple[int, str, int, int]]] = {
            tid: list(zip(d.stop_sequence, d.station_id, d.arr, d.dep))
            for tid, d in st.groupby("trip_id", sort=False)
        }
        # station_id -> list of (dep, trip_id, seq), sorted by departure
        self.boardings: dict[str, list[tuple[int, str, int]]] = {}
        # station_id -> list of (arr, trip_id, seq), sorted by arrival.
        # Needed by the backward (latest-departure) search.
        self.alightings: dict[str, list[tuple[int, str, int]]] = {}
        for tid, calls in self.trip_calls.items():
            for seq, sid, arr, dep in calls:
                self.boardings.setdefault(sid, []).append((int(dep), tid, int(seq)))
                self.alightings.setdefault(sid, []).append((int(arr), tid, int(seq)))
        for v in self.boardings.values():
            v.sort()
        for v in self.alightings.values():
            v.sort()

        self.world = world
        self.min_transfer = world.min_transfer_sec


def solve(
    c: Constraints,
    sw: SolverWorld,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> Solution | None:
    """Earliest arrival at c.destination. None if no feasible itinerary exists.

    Constraints enter differently by kind:
      avoid          -- graph-modifying: forbidden stations cannot be boarded,
                        alighted at, OR passed through, so a trip is abandoned
                        on reaching one.
      max_transfers  -- bounds the round index.
      arrive_before  -- prunes labels; monotone, so pruning is exact.
      depart_after   -- the search horizon.
    """
    if c.origin in c.avoid or c.destination in c.avoid:
        return None

    # A caller's max_transfers is a *constraint*, not a hint. Honour it or
    # raise -- never clamp it down to max_rounds and return a wrong answer.
    rounds = max_rounds if c.max_transfers is None else c.max_transfers + 1
    assert rounds <= HARD_CAP, (
        f"{rounds} rounds requested (max_transfers={c.max_transfers}), "
        f"HARD_CAP={HARD_CAP}"
    )
    rounds = max(1, rounds)
    deadline = INF if c.arrive_before is None else c.arrive_before

    # tau[k][station] = earliest arrival using at most k legs
    tau: list[dict[str, int]] = [{} for _ in range(rounds + 1)]
    parent: list[dict[str, tuple[str, str]]] = [{} for _ in range(rounds + 1)]
    tau[0][c.origin] = c.depart_after
    best: dict[str, int] = {c.origin: c.depart_after}
    marked = {c.origin}
    converged = False

    for k in range(1, rounds + 1):
        tau[k] = dict(tau[k - 1])
        parent[k] = dict(parent[k - 1])

        # Earliest boardable call of each trip, over all marked stations.
        board: dict[str, tuple[int, str]] = {}  # trip -> (seq, from_station)
        for s in marked:
            ready = best[s] + (0 if k == 1 else sw.min_transfer)
            for dep, tid, seq in sw.boardings.get(s, ()):
                if dep < ready:
                    continue
                if tid not in board or seq < board[tid][0]:
                    board[tid] = (seq, s)

        new_marked: set[str] = set()
        for tid, (b_seq, from_s) in board.items():
            for seq, sid, arr, _dep in sw.trip_calls[tid]:
                if seq <= b_seq:
                    continue
                if sid in c.avoid:
                    break  # cannot ride through a forbidden station
                if arr > deadline:
                    break  # arrivals along a trip are nondecreasing
                if arr < tau[k].get(sid, INF) and arr < best.get(sid, INF):
                    tau[k][sid] = arr
                    parent[k][sid] = (tid, from_s)
                    best[sid] = arr
                    new_marked.add(sid)

        marked = new_marked
        if not marked:
            converged = True  # no station improved: exact, whatever the cap
            break

    # Earliest arrival at the destination, over all rounds.
    hits = [(tau[k].get(c.destination, INF), k) for k in range(1, rounds + 1)]
    arrival, k = min(hits)
    if arrival >= INF:
        return None

    # Reconstruct. Walk parents back from the destination.
    legs: list[Leg] = []
    node, kk = c.destination, k
    while node != c.origin:
        while kk > 0 and node not in parent[kk]:
            kk -= 1
        if kk == 0:
            return None  # unreachable; should not happen
        tid, from_s = parent[kk][node]
        legs.append(Leg(tid, from_s, node))
        node, kk = from_s, kk - 1
    legs.reverse()

    calls = sw.trip_calls[legs[0].trip_id]
    departure = next(dep for _s, sid, _a, dep in calls if sid == legs[0].board)

    # When the user set max_transfers, the round bound IS the constraint, not a
    # truncation: exploring further would violate the request. The answer is
    # the constrained optimum, which is what we want.
    return Solution(legs, arrival, int(departure), len(legs) - 1, converged=converged)


# --- latest departure: the dual objective -----------------------------------
#
# "I need to be in Brno by 18:00 -- when do I leave?"
#
# Under earliest arrival, `arrive_before` can never bind: free.arrival IS the
# minimum arrival, so any tighter deadline is infeasible rather than tight.
# Under latest departure it is the constraint the query is *about*, and
# `depart_after` becomes the slack one. Same label-setting, run backwards.


def solve_latest(
    c: Constraints,
    sw: SolverWorld,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> Solution | None:
    """Latest departure from c.origin that still reaches c.destination by
    c.arrive_before. None if no feasible itinerary exists.

    sigma[k][s] = latest time one can be at s, with at most k legs remaining,
    and still arrive by the deadline.
    """
    assert c.arrive_before is not None, "latest departure needs a deadline"
    if c.origin in c.avoid or c.destination in c.avoid:
        return None

    rounds = max_rounds if c.max_transfers is None else c.max_transfers + 1
    assert rounds <= HARD_CAP, (
        f"{rounds} rounds requested (max_transfers={c.max_transfers}), "
        f"HARD_CAP={HARD_CAP}"
    )
    rounds = max(1, rounds)
    NEG = -1

    sigma: list[dict[str, int]] = [{} for _ in range(rounds + 1)]
    parent: list[dict[str, tuple[str, str]]] = [{} for _ in range(rounds + 1)]
    sigma[0][c.destination] = c.arrive_before
    best: dict[str, int] = {c.destination: c.arrive_before}
    marked = {c.destination}
    converged = False

    for k in range(1, rounds + 1):
        sigma[k] = dict(sigma[k - 1])
        parent[k] = dict(parent[k - 1])

        # Latest alighting call of each trip, over all marked stations.
        alight: dict[str, tuple[int, str]] = {}  # trip -> (seq, to_station)
        for s in marked:
            limit = best[s] - (0 if k == 1 else sw.min_transfer)
            for arr, tid, seq in sw.alightings.get(s, ()):
                if arr > limit:
                    break  # sorted by arrival
                if tid not in alight or seq > alight[tid][0]:
                    alight[tid] = (seq, s)

        new_marked: set[str] = set()
        for tid, (a_seq, to_s) in alight.items():
            for seq, sid, _arr, dep in reversed(sw.trip_calls[tid]):
                if seq >= a_seq:
                    continue
                if sid in c.avoid:
                    break  # cannot ride through a forbidden station
                if dep < c.depart_after:
                    break  # departures decrease walking backwards
                if dep > sigma[k].get(sid, NEG) and dep > best.get(sid, NEG):
                    sigma[k][sid] = dep
                    parent[k][sid] = (tid, to_s)
                    best[sid] = dep
                    new_marked.add(sid)

        marked = new_marked
        if not marked:
            converged = True
            break

    hits = [(sigma[k].get(c.origin, NEG), k) for k in range(1, rounds + 1)]
    departure, k = max(hits)
    if departure == NEG:
        return None

    legs: list[Leg] = []
    node, kk = c.origin, k
    while node != c.destination:
        while kk > 0 and node not in parent[kk]:
            kk -= 1
        if kk == 0:
            return None
        tid, to_s = parent[kk][node]
        legs.append(Leg(tid, node, to_s))
        node, kk = to_s, kk - 1

    calls = sw.trip_calls[legs[-1].trip_id]
    arrival = next(arr for _s, sid, arr, _d in calls if sid == legs[-1].alight)
    return Solution(legs, int(arrival), departure, len(legs) - 1, converged=converged)


# --- constraint evaluation, consuming replay()'s facts ---------------------


def violations(report, c: Constraints) -> dict[str, float]:
    """Per-constraint violation magnitude. Zero means satisfied.

    Only meaningful on a feasible report. Feasibility gates the reward.
    """
    assert report.feasible, "violations() on an infeasible itinerary"
    v: dict[str, float] = {}

    if c.arrive_before is not None:
        v["arrive_before"] = max(0, report.arrival - c.arrive_before) / 3600
    if c.max_transfers is not None:
        v["max_transfers"] = max(0, report.transfers - c.max_transfers)
    if c.avoid:
        v["avoid"] = len(c.avoid & set(report.stations_visited))
    v["depart_after"] = max(0, c.depart_after - report.departure) / 3600
    v["origin"] = 0.0 if report.stations_visited[0] == c.origin else 1.0
    v["destination"] = 0.0 if report.stations_visited[-1] == c.destination else 1.0
    return v


def satisfies(report, c: Constraints) -> bool:
    return report.feasible and not any(violations(report, c).values())