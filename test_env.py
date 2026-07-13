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


@case("retryable submit: a feasible itinerary is accepted")
def _(e):
    v = e.check_submit([Leg("T1", "P", "K"), Leg("T2", "K", "HB")], C(arrive_before=h(11)))
    assert v["feasible"] is True, v


@case("retryable submit: an infeasible itinerary is rejected with reasons")
def _(e):
    v = e.check_submit([Leg("T99", "P", "K")], C(arrive_before=h(11)))  # T99 doesn't exist
    assert v["feasible"] is False, v
    assert "reasons" in v or "error" in v, v


@case("retryable submit: feasible-but-violating is accepted with a note")
def _(e):
    v = e.check_submit([Leg("T1", "P", "K"), Leg("T2", "K", "HB")], C(arrive_before=h(5)))
    assert v["feasible"] is True, v
    assert v.get("violations"), v


@case("retryable submit: check_submit feasibility agrees with score()")
def _(e):
    for legs in (
        [Leg("T1", "P", "K"), Leg("T2", "K", "HB")],
        [Leg("T99", "P", "K")],
        [Leg("T1", "P", "K"), Leg("T2", "Pa", "HB")],
    ):
        cs = e.check_submit(legs, C(arrive_before=h(11)))["feasible"]
        sc = e.score(legs, C(arrive_before=h(11)), 0).feasible
        assert cs == sc, (legs, cs, sc)


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


# --- navigational distance signal -------------------------------------------


@case("signal: departures show km-to-dest for each terminus when dest is set")
def _(e):
    e.set_destination("HB")
    out = e.departures("P", "00:00")
    assert "here_km_to_dest" in out, out
    assert any("terminus_km_to_dest" in r for r in out["departures"]), out
    e.set_destination(None)


@case("signal: no distance fields when destination is unset")
def _(e):
    e.set_destination(None)
    out = e.departures("P", "00:00")
    assert "here_km_to_dest" not in out
    assert all("terminus_km_to_dest" not in r for r in out["departures"])


@case("signal: leg reports how far the alight station is from the destination")
def _(e):
    e.set_destination("HB")
    r = e.leg("T1", "P", "K")
    assert "to_km_to_dest" in r, r
    # K (lon 14.5) is closer to HB (15.6) than P (14.0) is: progress shows.
    assert r["to_km_to_dest"] < e.w.distance_km("P", "HB")
    e.set_destination(None)


# --- potential-based shaping ------------------------------------------------


@case("shaping: reaching the destination gives the full positive shaping")
def _(e):
    e.set_destination("HB")
    s = e._shaping([Leg("T1", "P", "K"), Leg("T2", "K", "HB")])  # ends AT HB
    from env import SHAPE_SCALE
    assert abs(s - SHAPE_SCALE) < 1e-9, s
    e.set_destination(None)


@case("shaping: measures the verified end, not the submitted end")
def _(e):
    e.set_destination("HB")
    from env import SHAPE_SCALE
    # A real connecting chain P->K (T1) then K->HB (T2) reaches HB: full credit.
    real = e._shaping([Leg("T1", "P", "K"), Leg("T2", "K", "HB")])
    assert abs(real - SHAPE_SCALE) < 1e-9, real
    # Same first leg, then a FABRICATED second leg (T99 doesn't exist): the
    # verified end is only K, not HB. Shaping reflects reaching K, not the
    # fabricated claim of HB. This is the blind-spot fix: inventing a leg to a
    # well-placed station earns nothing beyond where the real trains got you.
    faked = e._shaping([Leg("T1", "P", "K"), Leg("T99", "K", "HB")])
    assert faked < real, (faked, real)      # fabrication does not reach the goal
    # and it equals just the real first leg alone
    just_first = e._shaping([Leg("T1", "P", "K")])
    assert abs(faked - just_first) < 1e-9, (faked, just_first)
    e.set_destination(None)


@case("shaping: a fabricated first leg earns zero (no verified progress)")
def _(e):
    e.set_destination("HB")
    # If even the first leg is invented, the journey verifiably reaches nowhere:
    # the true position is still the origin, so no progress, no shaping.
    s = e._shaping([Leg("T99", "P", "HB")])
    assert s == 0.0, s
    e.set_destination(None)


@case("shaping: depends only on the verified endpoints, not path length")
def _(e):
    e.set_destination("HB")
    # T3 (P->K) and T1 (P->K) both really reach K. Shaping depends on the
    # verified end (K), not how the chain is written, so both give the same.
    a = e._shaping([Leg("T3", "P", "K")])
    b = e._shaping([Leg("T1", "P", "K")])
    assert abs(a - b) < 1e-9, (a, b)
    e.set_destination(None)


@case("shaping: ending farther from the goal is negative")
def _(e):
    e.set_destination("P")   # goal is the western end
    # T1 verifies P->K->Pa (a 3-stop trip). Board K, alight Pa: a real leg that
    # moves AWAY from P (Pa is east of K). Verified end Pa is farther from P
    # than origin K, so shaping is negative.
    s = e._shaping([Leg("T1", "K", "Pa")])
    assert s < 0, s
    e.set_destination(None)


@case("shaping: never lets an infeasible submission outrank a feasible one")
def _(e):
    from env import INFEASIBLE_BASE, INFEASIBLE_CREDIT, SHAPE_SCALE
    # The cap must hold for the theoretical worst case: max partial progress
    # (1.0) AND max shaping (SHAPE_SCALE) simultaneously. Assert the numeric
    # invariant the cap guarantees -- the infeasible ceiling stays below the
    # feasible floor (0.0) -- rather than hoping a fixture input hits the corner.
    uncapped_bonus = INFEASIBLE_CREDIT * 1.0 + SHAPE_SCALE
    capped_bonus = min(uncapped_bonus, INFEASIBLE_BASE - 0.05)
    ceiling = -INFEASIBLE_BASE + capped_bonus       # best infeasible, budget=0
    assert ceiling < 0.0, ceiling
    # and that the cap is actually doing work (uncapped would breach 0)
    assert -INFEASIBLE_BASE + uncapped_bonus >= 0.0, "cap is not needed -- retune"

    e.set_destination("HB")
    best_infeasible = e.score(
        [Leg("T1", "P", "K"), Leg("T2", "Pa", "HB")], C(arrive_before=h(11)), 0
    ).reward
    worst_feasible = e.score(
        [Leg("T1", "P", "K"), Leg("T2", "K", "HB")], C(arrive_before=h(5)), 0
    ).reward
    assert best_infeasible < worst_feasible, (best_infeasible, worst_feasible)
    e.set_destination(None)


@case("shaping: off when destination has no coordinates")
def _(e):
    e.set_destination("HB")
    # legs whose endpoints exist but pretend dest is coord-less: force via a
    # station id not in the coordinate table.
    s = e._shaping([Leg("T1", "P", "K")])
    e.set_destination("NOWHERE")
    s2 = e._shaping([Leg("T1", "P", "K")])
    assert s2 == 0.0, s2
    e.set_destination(None)


def main():
    tmp = Path(tempfile.mkdtemp())
    try:
        pd.DataFrame(
            ROWS, columns=["trip_id", "stop_sequence", "station_id", "arr", "dep"]
        ).to_parquet(tmp / "stop_times.parquet", index=False)
        ids = list(NAMES)
        comp = {i: 0 for i in ids}
        comp["C2"] = 1  # Čáslav město is stranded on a branch
        # Distinct coordinates along a rough west->east line so distance-to-dest
        # is meaningful. Longitude increases with the natural travel direction;
        # HB (Havlíčkův Brod) is the eastern destination in the shaping tests.
        lon = {"P": 14.0, "K": 14.5, "Pa": 15.0, "HB": 15.6, "C1": 14.6,
               "C2": 14.7, "PK": 14.3}
        lat = {s: 50.0 for s in ids}
        pd.DataFrame(
            {
                "station_id": ids,
                "stop_name": [NAMES[i] for i in ids],
                "stop_lat": [lat.get(i, 50.0) for i in ids],
                "stop_lon": [lon.get(i, 14.0) for i in ids],
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