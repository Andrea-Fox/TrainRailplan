"""
env.py -- the environment.

Used by three things, deliberately identical in all of them:
  * the headroom check (prompted model, no training)
  * SFT trace collection (rejection sampling)
  * GRPO rollouts

If the headroom check used a different harness, the number it produced would be
about the harness.

FORMALLY: a deterministic, episodic POMDP with static hidden state.
  hidden state  the frozen snapshot; the agent's actions never change it
  observation   a deterministic function of (call, snapshot)
  policy state  the context: request + full call/result history
  transition    deterministic; all stochasticity is in the policy
  reward        terminal only
  termination   submit(), or the call budget

Actions therefore serve exactly one purpose: acquiring information. The policy
learns a SEARCH STRATEGY. There is no control problem.

NOT A CMDP. The constraints (max_transfers, avoid, arrive_before) are properties
of the submitted itinerary, checked exactly, per episode, inside the scalar
reward. A CMDP constraint would be E_pi[C_i] <= 0 -- a property of the policy's
distribution over episodes, which is what a Lagrange multiplier prices. Adding
lambda later is a different object, not a reweighting of this one.

TOOLS. Local lookups only. There is no plan() and the solver is not reachable
from here. Search, backtracking, and stopping are the policy's job.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from replay import DAY, Leg, Report, World, hhmm, replay
from solve import Constraints, violations

# Terminal reward weights. Reported, not tuned into the noise.
W = {
    "arrive_before": 1.0,  # per hour late
    "max_transfers": 0.5,  # per transfer over budget
    "avoid": 1.0,  # per forbidden station visited
    "depart_after": 1.0,  # per hour early
    "origin": 1.0,
    "destination": 1.0,
}
TOOL_CALL_COST = 0.01  # terminal, over the count; GRPO has no per-step credit
MAX_CALLS = 30
DEPARTURES_LIMIT = 8  # context length is the real horizon bound

# Reward tiers, strictly ordered, independent of the tool-call budget (which is
# subtracted uniformly from all of them). Before this, never-submitting and
# submitting an infeasible itinerary both scored exactly -budget: the policy
# had NO incentive to keep searching rather than guess, and guessing is
# cheaper in tool calls, so GRPO would learn to guess early. Measured on Qwen:
# it does exactly this, submitting fabricated itineraries after 2-3 calls.
#
#   never submitted           < -GIVE_UP_PENALTY               (worst: gave up)
#   infeasible, no progress   = -INFEASIBLE_BASE                (fabricated)
#   infeasible, some progress -> -INFEASIBLE_BASE + INFEASIBLE_CREDIT  (near miss)
#   feasible, worst           = 0.0                              (strictly above)
#   feasible, best            = 1.0
GIVE_UP_PENALTY = 0.5
INFEASIBLE_BASE = 0.3
INFEASIBLE_CREDIT = 0.2  # -INFEASIBLE_BASE + INFEASIBLE_CREDIT < 0.0, always

# Potential-based shaping (Ng, Harada & Russell 1999). Phi(s) = -distance(s,dest).
# The shaping over a chain of legs telescopes to Phi(alight_last) - Phi(origin),
# i.e. it depends ONLY on how far the journey's endpoints are from the goal, not
# on the path taken. This is what makes it unhackable: a loop returns to where it
# started and nets zero shaping; a dead-end branch that ends far from the goal
# nets NEGATIVE shaping. Only sustained progress toward the destination pays, and
# only measured at the endpoints -- the agent cannot farm it by inching closer
# and back. Scaled small (SHAPE_SCALE) so it guides without overturning the tier
# ordering: it never lets an infeasible submission outrank a feasible one.
SHAPE_SCALE = 0.15  # max shaping magnitude, as a fraction of a normalized trip


def partial_progress(legs: list[Leg], world: World) -> float:
    """Fraction of the proposed chain that validates before the first break.

    Bounded in [0, 1). If every leg validated, replay() would have returned
    feasible=True and this function is never reached for that case -- if it
    ever returns something >= (len(legs)-1)/len(legs) that is a sign this
    check has drifted out of sync with replay()'s.

    This grades HOW the itinerary is wrong, not just THAT it is wrong: a trip
    that doesn't exist at all (hallucination) breaks at index 0 and scores the
    infeasible floor; an otherwise-correct chain broken only by a too-tight
    final transfer gets most of the credit. Mirrors replay()'s checks but
    stops at the first failure rather than collecting every reason -- this is
    a reward signal, not a diagnostic.
    """
    if not legs:
        return 0.0
    prev_alight = prev_arr = None

    for i, leg in enumerate(legs):
        if not world.has_trip(leg.trip_id):
            return i / len(legs)
        calls = world.calls(leg.trip_id)
        if leg.board not in calls or leg.alight not in calls:
            return i / len(legs)
        if leg.board == leg.alight:
            # Redundant with the b_seq >= a_seq check below (equal indices
            # satisfy >=), kept for clarity: this is the case a degenerate
            # policy actually produces (leg(X, X)), not a hypothetical.
            return i / len(legs)

        b_seq, _, b_dep = calls[leg.board]
        a_seq, a_arr, _ = calls[leg.alight]
        if b_seq >= a_seq:
            return i / len(legs)

        if prev_alight is not None:
            if leg.board != prev_alight:
                return i / len(legs)
            if b_dep < prev_arr + world.min_transfer_sec:
                return i / len(legs)

        prev_alight, prev_arr = leg.alight, a_arr

    # Every leg validated individually and chained -- replay() disagreeing
    # means this helper is out of sync with it. Don't claim full credit for a
    # disagreement; fall back to "one short of complete" rather than crash a
    # rollout on a reward-shaping bug.
    return (len(legs) - 1) / len(legs)


def fold(s: str) -> str:
    """Diacritic- and case-insensitive key. 'Zatec' should find 'Žatec' --
    a policy typing ASCII is not making a semantic error."""
    n = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in n if not unicodedata.combining(c)).strip()


@dataclass
class Call:
    tool: str
    args: dict
    result: dict

    def render(self) -> str:
        return json.dumps(self.result, ensure_ascii=False)


@dataclass
class Outcome:
    reward: float
    feasible: bool
    submitted: bool
    n_calls: int
    report: Report | None = None
    violations: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    progress: float | None = None  # infeasible submissions only; see partial_progress


class Env:
    def __init__(self, world: World, snapshot: str | Path):
        self.w = world
        snap = Path(snapshot)

        st = pd.read_parquet(snap / "stop_times.parquet")
        st = st.sort_values(["trip_id", "stop_sequence"])

        # station_id -> [(dep, trip_id, seq)], sorted.
        #
        # The LAST call of a trip is excluded. You cannot board a train at its
        # terminus, and leg() would reject any such boarding -- so offering it
        # in departures() is pure noise: a wasted tool call, and a lesson that
        # the tool's output cannot be trusted.
        self.boardings: dict[str, list[tuple[int, str, int]]] = {}
        for tid, d in st.groupby("trip_id", sort=False):
            rows = list(zip(d.stop_sequence, d.station_id, d.arr, d.dep))
            for seq, sid, _arr, dep in rows[:-1]:
                self.boardings.setdefault(sid, []).append((int(dep), tid, int(seq)))
        for v in self.boardings.values():
            v.sort()

        stations = pd.read_parquet(snap / "stations.parquet")
        self.by_fold: dict[str, list[str]] = {}
        for sid, name in zip(stations.station_id, stations.stop_name):
            self.by_fold.setdefault(fold(name), []).append(sid)
        self.names = dict(zip(stations.station_id, stations.stop_name))
        self.component = dict(zip(stations.station_id, stations.component))

        # Boardable departures per station. The ranking signal for find_station:
        # when a query matches many stations, the one people mean is the one
        # trains actually leave from.
        self.service = {s: len(v) for s, v in self.boardings.items()}

        # Episode destination, for the km-to-destination navigational signal.
        # Set per episode via set_destination(); None means the signal is off
        # (e.g. during unit tests that don't exercise navigation).
        self._dest: str | None = None

    def set_destination(self, dest_id: str | None) -> None:
        """Call at episode start. The distance signal in departures() and leg()
        is measured to this station. The destination is stated in the request,
        so surfacing distance to it is fair -- it tells the policy nothing it
        was not already given, only does the arithmetic it cannot do itself."""
        self._dest = dest_id

    # --- tools ------------------------------------------------------------

    def find_station(self, query: str, limit: int = 5) -> dict:
        """Substring match on the folded name, ranked.

        RANKING MATTERS. Czechia has ~30 stations whose name contains "Praha".
        Ranked by name length, `find_station("praha")` returns Praha-Krc,
        Praha-Eden, Praha-Kyje, Praha-Kbely, Praha-Liben -- and truncates before
        Praha hlavni nadrazi, the main station of the country. Every instance
        with a Praha endpoint would then be unsolvable through the tool.

        So: exact match, then prefix, then substring; within each tier, by how
        many trains actually depart. When a query is ambiguous, the station the
        traveller means is the one with the service.

        Ambiguity is SURFACED, never resolved. Caslav and Caslav m. n. are two
        stations in one town, sitting in different connected components. A tool
        that guessed would hand the policy an instance that is infeasible for
        reasons it cannot observe.
        """
        q = fold(query)
        if not q:
            return {"error": "empty query"}

        exact, prefix, sub = [], [], []
        for key, sids in self.by_fold.items():
            if key == q:
                exact.extend(sids)
            elif key.startswith(q):
                prefix.extend(sids)
            elif q in key:
                sub.extend(sids)

        by_service = lambda s: -self.service.get(s, 0)
        hits = (
            sorted(exact, key=by_service)
            + sorted(prefix, key=by_service)
            + sorted(sub, key=by_service)
        )

        if not hits:
            return {"query": query, "matches": [], "hint": "no station matches"}
        return {
            "query": query,
            "matches": [
                {"station_id": s, "name": self.names[s]} for s in hits[:limit]
            ],
            "n_matches": len(hits),
            "truncated": len(hits) > limit,
        }

    def departures(self, station_id: str, after: str, limit: int = DEPARTURES_LIMIT) -> dict:
        """Next departures from a station. `after` is HH:MM, possibly past 24."""
        if station_id not in self.names:
            return {"error": f"unknown station_id {station_id!r}"}
        t = parse_hhmm(after)
        if t is None:
            return {"error": f"bad time {after!r}, expected HH:MM"}

        rows = []
        for dep, tid, seq in self.boardings.get(station_id, ()):
            if dep < t:
                continue
            calls = self.w.calls(tid)
            last = max(calls.items(), key=lambda kv: kv[1][0])
            row = {
                "trip_id": tid,
                "departs": hhmm(dep),
                "towards": self.names[last[0]],
                "final_arrival": hhmm(last[1][1]),
            }
            # Navigational signal: how far this train's terminus is from the
            # destination. Lower means the train heads goalward -- a heuristic
            # to rank candidates, NOT a guarantee (a nearer terminus may be a
            # dead-end branch with no onward service).
            if self._dest is not None:
                d = self.w.distance_km(last[0], self._dest)
                if d is not None:
                    row["terminus_km_to_dest"] = round(d, 1)
            rows.append(row)
            if len(rows) >= limit:
                break

        out = {
            "station": self.names[station_id],
            "after": after,
            "departures": rows,
            "truncated": len(rows) >= limit,
        }
        if self._dest is not None:
            here = self.w.distance_km(station_id, self._dest)
            if here is not None:
                out["here_km_to_dest"] = round(here, 1)
        return out

    def leg(self, trip_id: str, board: str, alight: str) -> dict:
        """Times for riding one trip between two stations, or why you can't."""
        if not self.w.has_trip(trip_id):
            return {"error": f"trip {trip_id!r} does not run today"}
        calls = self.w.calls(trip_id)
        if board not in calls:
            return {"error": f"trip {trip_id} does not call at {board!r}"}
        if alight not in calls:
            return {"error": f"trip {trip_id} does not call at {alight!r}"}

        b_seq, _, b_dep = calls[board]
        a_seq, a_arr, _ = calls[alight]
        if b_seq >= a_seq:
            return {
                "error": f"trip {trip_id} runs the other way",
                "hint": f"it calls at {self.names[alight]} before {self.names[board]}",
            }
        via = sorted(
            (s for s, (seq, _, _) in calls.items() if b_seq < seq < a_seq),
            key=lambda s: calls[s][0],
        )
        result = {
            "trip_id": trip_id,
            "departs": hhmm(b_dep),
            "arrives": hhmm(a_arr),
            "from": self.names[board],
            "to": self.names[alight],
            "calls_at": [self.names[s] for s in via],
        }
        # Did this hop get closer? Show the alight station's distance to the
        # destination so the policy can judge progress against where it was.
        if self._dest is not None:
            d = self.w.distance_km(alight, self._dest)
            if d is not None:
                result["to_km_to_dest"] = round(d, 1)
        return result

    # --- episode ----------------------------------------------------------

    def step(self, tool: str, args: dict) -> dict:
        """A malformed call is an OBSERVATION, not a crash. The policy must be
        able to recover from its own mistakes -- that is most of the task."""
        try:
            if tool == "find_station":
                return self.find_station(**args)
            if tool == "departures":
                return self.departures(**args)
            if tool == "leg":
                return self.leg(**args)
            return {"error": f"unknown tool {tool!r}"}
        except TypeError as e:
            return {"error": f"bad arguments for {tool}: {e}"}

    def _shaping(self, legs: list[Leg]) -> float:
        """Telescoped potential-based shaping, in [-SHAPE_SCALE, +SHAPE_SCALE].

        Phi(s) = -distance(s, dest). The per-leg shaping sums to
        Phi(alight_last) - Phi(board_first) = dist(origin,dest) - dist(end,dest),
        normalized by dist(origin,dest) so it is scale-free across journeys:

            +SHAPE_SCALE  the chain ends AT the destination (full progress)
             0            it ends as far from the goal as it began
            -SHAPE_SCALE  it ends twice as far (wandered backwards), clipped

        Depends only on the endpoints, so loops and detours net nothing. Returns
        0 when the destination or an endpoint lacks coordinates -- shaping is a
        guide, never a requirement.
        """
        if self._dest is None or not legs:
            return 0.0
        origin, end = legs[0].board, legs[-1].alight
        d0 = self.w.distance_km(origin, self._dest)
        d1 = self.w.distance_km(end, self._dest)
        if d0 is None or d1 is None or d0 < 1e-6:
            return 0.0
        frac = (d0 - d1) / d0            # 1 at the goal, 0 no progress, <0 backwards
        return SHAPE_SCALE * max(-1.0, min(1.0, frac))

    def score(self, legs: list[Leg] | None, c: Constraints, n_calls: int) -> Outcome:
        """Terminal reward, in strict tiers (independent of the budget, which
        is subtracted uniformly from all of them):

            never submitted            < infeasible, no progress
            infeasible, no progress     = -INFEASIBLE_BASE
            infeasible, some progress  -> -INFEASIBLE_BASE + INFEASIBLE_CREDIT
            feasible, any violations    = 0.0
            feasible, no violations     = 1.0

        Feasibility still gates the top tier: an impossible itinerary can
        never outscore a feasible one, however close it got. But within
        'impossible', HOW wrong it is now matters -- a fabricated trip ID
        scores worse than a real chain broken by a too-tight transfer -- and
        giving up always scores worse than submitting anything at all. Before
        this, giving up and fabricating an answer scored identically, so the
        policy had no reason to keep searching instead of guessing.
        """
        budget = TOOL_CALL_COST * n_calls

        if not legs:
            return Outcome(
                -budget - GIVE_UP_PENALTY, False, False, n_calls,
                reasons=["never submitted"],
            )

        rep = replay(legs, self.w)
        if not rep.feasible:
            progress = partial_progress(legs, self.w)
            # Two graded signals on a failed submission, and they answer
            # different questions. partial_progress: how much of the CHAIN
            # validated (did the trains connect). shaping: how far the journey
            # got GEOGRAPHICALLY toward the goal. A near-miss that also ended up
            # close to the destination scores highest within the infeasible tier;
            # a fabricated chain going nowhere scores lowest. Both are bounded so
            # the tier as a whole still sits strictly below any feasible route.
            shaped = self._shaping(legs)
            bonus = INFEASIBLE_CREDIT * progress + shaped
            # The infeasible tier must stay strictly below the feasible floor
            # (0.0). progress<=1 and shaped<=SHAPE_SCALE could sum past
            # INFEASIBLE_BASE, so clip the bonus to keep the ceiling negative.
            bonus = min(bonus, INFEASIBLE_BASE - 0.05)
            reward = -budget - INFEASIBLE_BASE + bonus
            return Outcome(
                reward, False, True, n_calls, rep,
                reasons=rep.reasons, progress=progress,
            )

        v = violations(rep, c)
        penalty = sum(W[k] * val for k, val in v.items())
        reward = max(0.0, 1.0 - penalty) - budget
        return Outcome(reward, True, True, n_calls, rep, violations=v)


def parse_hhmm(s: str) -> int | None:
    try:
        hh, mm = s.strip().split(":")
        h, m = int(hh), int(mm)
    except (ValueError, AttributeError):
        return None
    if not (0 <= m < 60 and 0 <= h < 48):
        return None
    return h * 3600 + m * 60


TOOLS_SPEC = """
find_station(query: str)
    Look up stations by name. Returns candidates; several may match, and they
    may be different stations in the same town. Disambiguate before using one.

departures(station_id: str, after: str)
    Next departures from a station, after a HH:MM time. Times may exceed 24:00
    for trains running past midnight.

leg(trip_id: str, board: str, alight: str)
    Ride one train between two stations. Returns departure and arrival times,
    and the stations it calls at on the way. Errors if the train does not run
    that way.

submit(legs: [{trip_id, board, alight}, ...])
    Propose the itinerary and end the episode. Consecutive legs must chain:
    you must alight where you next board, with at least 3 minutes to change.
""".strip()