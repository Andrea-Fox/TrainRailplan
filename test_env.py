"""
Tests for env.py. Run: python test_env.py

Two things under test:
  * tools return OBSERVATIONS on bad input, never exceptions. Recovering from
    its own malformed calls is most of what the policy has to learn.
  * the reward has the shape we claim: feasibility gates, violations grade.
"""

import shutil
import tempfile
from pathlib import Path

import pandas as pd

from env import (
    GIVE_UP_PENALTY,
    INFEASIBLE_BASE,
    INFEASIBLE_CREDIT,
    TOOL_CALL_COST,
    Env,
    fold,
    parse_hhmm,
    partial_progress,
)
from replay import Leg, World
from solve import Constraints


def h(hh, mm=0):
    return hh * 3600 + mm * 60


#   T1: Praha 08:00 -> Kolin 09:00/09:02 -> Pardubice 09:30
#   T2: Kolin 09:20 -> HavlBrod 10:20
#   T3: Praha 07:00 -> Kolin 07:45          (terminates at Kolin)
#   T4: PKrc 08:30 -> Kolin 09:10           (one train from a suburban halt)
ROWS = [
    ("T1", 1, "P", h(8), h(8)),
    ("T1", 2, "K", h(9), h(9, 2)),
    ("T1", 3, "Pa", h(9, 30), h(9, 30)),
    ("T2", 1, "K", h(9, 20), h(9, 20)),
    ("T2", 2, "HB", h(10, 20), h(10, 20)),
    ("T3", 1, "P", h(7), h(7)),
    ("T3", 2, "K", h(7, 45), h(7, 45)),
    ("T4", 1, "PK", h(8, 30), h(8, 30)),
    ("T4", 2, "K", h(9, 10), h(9, 10)),
]
NAMES = {
    "P": "Praha hlavní nádraží",
    "PK": "Praha-Krč",
    "PE": "Praha-Eden",
    "K": "Kolín",
    "Pa": "Pardubice hlavní nádraží",
    "HB": "Havlíčkův Brod",
    "C1": "Čáslav",
    "C2": "Čáslav město",
}

CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn

    return deco


# --- find_station -----------------------------------------------------------


@case("fold: diacritics and case are not semantic errors")
def _(e):
    assert fold("Žatec") == fold("zatec") == "zatec"
    assert fold("Ústí nad Labem") == "usti nad labem"


@case("find_station: ASCII query finds a diacritic name")
def _(e):
    r = e.find_station("kolin")
    assert [m["station_id"] for m in r["matches"]] == ["K"], r


@case("find_station: exact match ranks above substring")
def _(e):
    r = e.find_station("Čáslav")
    ids = [m["station_id"] for m in r["matches"]]
    assert ids[0] == "C1" and "C2" in ids, ids


@case("find_station: an ambiguous query surfaces the station with the service")
def _(e):
    # Ranked by name length, 'praha' returns Praha-Eden and Praha-Krc and
    # truncates before the main station. Rank by departures instead.
    r = e.find_station("praha", limit=2)
    assert r["matches"][0]["station_id"] == "P", r["matches"]
    assert r["n_matches"] == 3 and r["truncated"]


@case("find_station: prefix beats bare substring")
def _(e):
    # 'Kolin' is a substring of nothing else here, but the tiering must hold
    # when a longer name merely contains the query.
    r = e.find_station("kol")
    assert r["matches"][0]["station_id"] == "K"


@case("find_station: ambiguity is surfaced, not resolved")
def _(e):
    # Two stations, one town, different components. Guessing on the policy's
    # behalf would hand it an instance that is infeasible for invisible reasons.
    r = e.find_station("caslav")
    assert len(r["matches"]) == 2, r
    assert e.component["C1"] != e.component["C2"]


@case("departures: a trip terminating here is not boardable")
def _(e):
    # T3 ends at Kolin. Offering it as a departure from Kolin wastes a tool
    # call and teaches the policy that departures() output cannot be trusted.
    r = e.departures("K", "00:00", limit=20)
    assert "T3" not in [d["trip_id"] for d in r["departures"]], r["departures"]
    assert "T4" not in [d["trip_id"] for d in r["departures"]]  # T4 ends here too


@case("departures: everything offered is actually rideable")
def _(e):
    for sid in e.names:
        for d in e.departures(sid, "00:00", limit=20).get("departures", []):
            calls = e.w.calls(d["trip_id"])
            board_seq = calls[sid][0]
            assert any(seq > board_seq for seq, _, _ in calls.values()), (
                f"{d['trip_id']} offered at {e.names[sid]} but goes nowhere"
            )


@case("find_station: no match returns a hint, not an error")
def _(e):
    r = e.find_station("Atlantis")
    assert r["matches"] == [] and "hint" in r


@case("find_station: empty query is an error observation")
def _(e):
    assert "error" in e.find_station("   ")


# --- departures -------------------------------------------------------------


@case("departures: respects `after`")
def _(e):
    r = e.departures("P", "07:30")
    assert [d["trip_id"] for d in r["departures"]] == ["T1"], r


@case("departures: reports where the train is going")
def _(e):
    r = e.departures("P", "06:00")
    first = r["departures"][0]
    assert first["trip_id"] == "T3" and first["towards"] == "Kolín"


@case("departures: limit truncates and says so")
def _(e):
    r = e.departures("P", "00:00", limit=1)
    assert len(r["departures"]) == 1 and r["truncated"]


@case("departures: unknown station is an observation")
def _(e):
    assert "error" in e.departures("nope", "08:00")


@case("departures: malformed time is an observation")
def _(e):
    assert "error" in e.departures("P", "8am")
    assert "error" in e.departures("P", "25:99")


@case("parse_hhmm: extended times past midnight are valid")
def _(e):
    assert parse_hhmm("24:20") == h(24, 20)
    assert parse_hhmm("26:14") == h(26, 14)
    assert parse_hhmm("48:00") is None


# --- leg --------------------------------------------------------------------


@case("leg: returns times and intermediate calls")
def _(e):
    r = e.leg("T1", "P", "Pa")
    assert r["departs"] == "08:00" and r["arrives"] == "09:30"
    assert r["calls_at"] == ["Kolín"]


@case("leg: wrong direction explains itself")
def _(e):
    r = e.leg("T1", "Pa", "P")
    assert "error" in r and "other way" in r["error"]
    assert "hint" in r


@case("leg: unknown trip is an observation")
def _(e):
    assert "error" in e.leg("T99", "P", "K")


@case("leg: station not on the trip is an observation")
def _(e):
    assert "error" in e.leg("T1", "P", "HB")


# --- step -------------------------------------------------------------------


@case("step: unknown tool does not crash")
def _(e):
    assert "error" in e.step("plan", {"from": "P", "to": "HB"})


@case("step: wrong arguments do not crash")
def _(e):
    r = e.step("departures", {"station": "P"})
    assert "error" in r and "bad arguments" in r["error"]


# --- reward -----------------------------------------------------------------


def C(**kw):
    return Constraints("P", "HB", **kw)


@case("reward: giving up scores worse than submitting anything")
def _(e):
    give_up = e.score(None, C(arrive_before=h(11)), n_calls=5)
    # Fabricate a trip that doesn't exist -- the worst possible submission.
    fabricated = e.score([Leg("T99", "P", "K")], C(arrive_before=h(11)), n_calls=5)
    assert give_up.reward < fabricated.reward, (give_up.reward, fabricated.reward)


@case("reward: never submitting pays the budget plus the give-up penalty")
def _(e):
    o = e.score(None, C(arrive_before=h(11)), n_calls=5)
    assert not o.submitted
    assert abs(o.reward - (-5 * TOOL_CALL_COST - GIVE_UP_PENALTY)) < 1e-9


@case("reward: feasibility gates -- an impossible itinerary cannot beat a feasible one")
def _(e):
    # T1 arrives Kolin 09:00; T2 departs 09:20. Fine. But T1 then T1 is not.
    o = e.score([Leg("T1", "P", "K"), Leg("T1", "K", "Pa")], C(arrive_before=h(11)), 4)
    assert o.submitted and not o.feasible
    assert o.reasons
    worst_feasible = -4 * TOOL_CALL_COST  # feasible reward floors at 0.0 - budget
    assert o.reward < worst_feasible, (o.reward, worst_feasible)


@case("reward: infeasible submissions are graded by how far they got")
def _(e):
    # Fabricated trip: breaks at index 0 (T99 doesn't exist).
    fabricated = e.score([Leg("T99", "P", "K")], C(arrive_before=h(11)), 4)
    # Valid first leg, then a leg that references a station T2 never calls at:
    # breaks at index 1, so it got further than the fabricated one.
    near_miss = e.score([Leg("T1", "P", "K"), Leg("T2", "P", "HB")], C(arrive_before=h(11)), 4)
    assert fabricated.progress == 0.0, fabricated.progress
    assert near_miss.progress == 0.5, near_miss.progress
    assert near_miss.reward > fabricated.reward, (near_miss.reward, fabricated.reward)


@case("partial_progress: fabricated trip breaks at index 0")
def _(e):
    assert partial_progress([Leg("T99", "P", "K")], e.w) == 0.0


@case("partial_progress: board==alight breaks at that index")
def _(e):
    assert partial_progress([Leg("T1", "K", "K")], e.w) == 0.0


@case("partial_progress: wrong direction breaks at that index")
def _(e):
    assert partial_progress([Leg("T1", "Pa", "P")], e.w) == 0.0


@case("partial_progress: a good first leg counts before a bad second one")
def _(e):
    # Leg 0 is genuinely valid; leg 1 references a station T2 doesn't call at.
    p = partial_progress([Leg("T1", "P", "K"), Leg("T2", "P", "HB")], e.w)
    assert p == 0.5, p


@case("partial_progress: a tight transfer breaks only at the chain check")
def _(e):
    # Leg 0 valid. Leg 1's stations are individually fine on T2, but chaining
    # requires boarding where leg 0 alighted -- here it does (K), so instead
    # break it on timing: reuse T3 which departs before min_transfer allows.
    p = partial_progress([Leg("T4", "PK", "K"), Leg("T1", "P", "Pa")], e.w)
    assert p == 0.5, p  # leg 1 doesn't board where leg 0 alighted


@case("reward: a perfect itinerary scores 1 minus the budget")
def _(e):
    legs = [Leg("T1", "P", "K"), Leg("T2", "K", "HB")]
    o = e.score(legs, C(arrive_before=h(11)), n_calls=6)
    assert o.feasible and not any(o.violations.values()), o.violations
    assert abs(o.reward - (1.0 - 6 * TOOL_CALL_COST)) < 1e-9


@case("reward: feasible but late gets partial credit")
def _(e):
    legs = [Leg("T1", "P", "K"), Leg("T2", "K", "HB")]  # arrives 10:20
    o = e.score(legs, C(arrive_before=h(10)), n_calls=6)
    assert o.feasible
    assert abs(o.violations["arrive_before"] - 20 / 60) < 1e-9
    assert 0 < o.reward < 1


@case("reward: grades violation magnitude, not just presence")
def _(e):
    legs = [Leg("T1", "P", "K"), Leg("T2", "K", "HB")]
    mild = e.score(legs, C(arrive_before=h(10, 10)), 6).reward
    bad = e.score(legs, C(arrive_before=h(9)), 6).reward
    assert mild > bad, (mild, bad)


@case("reward: floors at zero before the budget, never rewarding a bad answer")
def _(e):
    legs = [Leg("T1", "P", "K"), Leg("T2", "K", "HB")]
    o = e.score(legs, C(arrive_before=h(5)), n_calls=0)  # 5h20m late
    assert o.reward == 0.0, o.reward


@case("reward: the call budget is terminal, over the count")
def _(e):
    legs = [Leg("T1", "P", "K"), Leg("T2", "K", "HB")]
    a = e.score(legs, C(arrive_before=h(11)), n_calls=2).reward
    b = e.score(legs, C(arrive_before=h(11)), n_calls=12).reward
    assert abs((a - b) - 10 * TOOL_CALL_COST) < 1e-9


@case("reward: violating avoid is caught on pass-through, not just transfer")
def _(e):
    # T1 passes through Kolín without a change being made there.
    o = e.score([Leg("T1", "P", "Pa")], Constraints("P", "Pa", avoid=frozenset({"K"})), 3)
    assert o.feasible and o.violations["avoid"] == 1
    assert o.reward < 1


@case("reward: strict tier ordering holds end to end")
def _(e):
    give_up = e.score(None, C(arrive_before=h(11)), 4).reward
    fabricated = e.score([Leg("T99", "P", "K")], C(arrive_before=h(11)), 4).reward
    near_miss = e.score(
        [Leg("T1", "P", "K"), Leg("T2", "Pa", "HB")], C(arrive_before=h(11)), 4
    ).reward
    worst_feasible = e.score(
        [Leg("T1", "P", "K"), Leg("T2", "K", "HB")], C(arrive_before=h(5)), 4
    ).reward
    best_feasible = e.score(
        [Leg("T1", "P", "K"), Leg("T2", "K", "HB")], C(arrive_before=h(11)), 4
    ).reward
    assert give_up < fabricated < near_miss < worst_feasible < best_feasible, (
        give_up, fabricated, near_miss, worst_feasible, best_feasible
    )


def main():
    tmp = Path(tempfile.mkdtemp())
    try:
        pd.DataFrame(
            ROWS, columns=["trip_id", "stop_sequence", "station_id", "arr", "dep"]
        ).to_parquet(tmp / "stop_times.parquet", index=False)
        ids = list(NAMES)
        comp = {i: 0 for i in ids}
        comp["C2"] = 1  # Čáslav město is stranded on a branch
        pd.DataFrame(
            {
                "station_id": ids,
                "stop_name": [NAMES[i] for i in ids],
                "stop_lat": [50.0] * len(ids),
                "stop_lon": [14.0] * len(ids),
                "geolocatable": [True] * len(ids),
                "n_stop_ids": [1] * len(ids),
                "component": [comp[i] for i in ids],
            }
        ).to_parquet(tmp / "stations.parquet", index=False)

        e = Env(World(tmp), tmp)
        failed = 0
        for name, fn in CASES:
            try:
                fn(e)
                print(f"  pass  {name}")
            except AssertionError as ex:
                failed += 1
                print(f"  FAIL  {name}\n        {ex}")
        print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
        return 1 if failed else 0
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    raise SystemExit(main())