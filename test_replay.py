"""
Adversarial tests for replay(). Run: python test_replay.py

The verifier is the reward. Every test here is an itinerary a policy could
plausibly submit, and each one must be judged correctly. A verifier that only
ever sees valid input is untested.
"""

import shutil
import tempfile
from pathlib import Path

import pandas as pd

from replay import Leg, World, replay

# ---------------------------------------------------------------------------
# A toy world.
#
#   T1: Praha 08:00 -> Kolin 09:00 -> Pardubice 09:30
#   T2: Kolin 09:02 -> Havlickuv Brod 10:00        (2 min after T1: too tight)
#   T3: Kolin 09:20 -> Havlickuv Brod 10:20        (18 min: fine)
#   T4: Pardubice 09:35 -> Kolin 10:05             (backwards, for direction test)
#   T5: Praha 23:40 -> Kolin 24:20                 (crosses midnight)
# ---------------------------------------------------------------------------
def h(hh, mm=0):
    return hh * 3600 + mm * 60


ROWS = [
    ("T1", 1, "Praha",   h(8),        h(8)),
    ("T1", 2, "Kolin",   h(9),        h(9, 2)),
    ("T1", 3, "Pardubice", h(9, 30),  h(9, 30)),
    ("T2", 1, "Kolin",   h(9, 2),     h(9, 2)),
    ("T2", 2, "HavlBrod", h(10),      h(10)),
    ("T3", 1, "Kolin",   h(9, 20),    h(9, 20)),
    ("T3", 2, "HavlBrod", h(10, 20),  h(10, 20)),
    ("T4", 1, "Pardubice", h(9, 35),  h(9, 35)),
    ("T4", 2, "Kolin",   h(10, 5),    h(10, 5)),
    ("T5", 1, "Praha",   h(23, 40),   h(23, 40)),
    ("T5", 2, "Kolin",   h(24, 20),   h(24, 20)),
]

STATIONS = ["Praha", "Kolin", "Pardubice", "HavlBrod", "Brno"]  # Brno: unserved


def build_world(tmp: Path, min_transfer=180) -> World:
    pd.DataFrame(
        ROWS, columns=["trip_id", "stop_sequence", "station_id", "arr", "dep"]
    ).to_parquet(tmp / "stop_times.parquet", index=False)
    pd.DataFrame(
        {
            "station_id": STATIONS,
            "stop_name": STATIONS,
            "stop_lat": [50.0, 50.0, 50.0, 49.6, 49.2],
            "stop_lon": [14.4, 15.2, 15.8, 15.6, 16.6],
            "geolocatable": [True] * 5,
            "n_stop_ids": [1] * 5,
            "component": [0, 0, 0, 0, 1],
        }
    ).to_parquet(tmp / "stations.parquet", index=False)
    return World(tmp, min_transfer_sec=min_transfer)


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn

    return deco


# --- must be accepted ------------------------------------------------------


@case("direct leg")
def _(w):
    r = replay([Leg("T1", "Praha", "Pardubice")], w)
    assert r, r.reasons
    assert r.departure == h(8) and r.arrival == h(9, 30)
    assert r.transfers == 0
    assert r.stations_visited == ["Praha", "Kolin", "Pardubice"]


@case("valid transfer with slack")
def _(w):
    r = replay([Leg("T1", "Praha", "Kolin"), Leg("T3", "Kolin", "HavlBrod")], w)
    assert r, r.reasons
    assert r.transfers == 1
    assert r.arrival == h(10, 20)
    assert r.stations_visited == ["Praha", "Kolin", "HavlBrod"]


@case("partial leg: board mid-trip")
def _(w):
    r = replay([Leg("T1", "Kolin", "Pardubice")], w)
    assert r, r.reasons
    assert r.departure == h(9, 2)


@case("midnight crossing is not a violation")
def _(w):
    r = replay([Leg("T5", "Praha", "Kolin")], w)
    assert r, r.reasons
    assert r.arrival == h(24, 20)
    assert r.duration == 40 * 60


# --- must be rejected ------------------------------------------------------


@case("empty itinerary")
def _(w):
    r = replay([], w)
    assert not r and "empty" in r.reasons[0]


@case("nonexistent trip")
def _(w):
    r = replay([Leg("T99", "Praha", "Kolin")], w)
    assert not r and "does not run" in r.reasons[0]


@case("trip does not call at station")
def _(w):
    r = replay([Leg("T1", "Praha", "HavlBrod")], w)
    assert not r and "does not call" in r.reasons[0]


@case("impossible transfer: 2 minutes, needs 3")
def _(w):
    r = replay([Leg("T1", "Praha", "Kolin"), Leg("T2", "Kolin", "HavlBrod")], w)
    assert not r, "the 3-minute impossible transfer must be caught"
    assert "to change at" in r.reasons[0]


@case("transfer exactly at the minimum is allowed")
def _(w):
    r = replay(
        [Leg("T1", "Praha", "Kolin"), Leg("T2", "Kolin", "HavlBrod")],
        w,
        min_transfer_sec=120,
    )
    assert r, r.reasons  # T1 arrives 09:00, T2 departs 09:02: exactly 120s


@case("wrong direction: alight before board")
def _(w):
    r = replay([Leg("T1", "Pardubice", "Praha")], w)
    assert not r and "not the direction requested" in r.reasons[0]


@case("teleport: legs do not chain")
def _(w):
    r = replay([Leg("T1", "Praha", "Kolin"), Leg("T4", "Pardubice", "Kolin")], w)
    assert not r and "alight at" in r.reasons[0]


@case("board equals alight")
def _(w):
    r = replay([Leg("T1", "Kolin", "Kolin")], w)
    assert not r and "boards and alights" in r.reasons[0]


@case("time travel: second leg departs before first arrives")
def _(w):
    # T4 leaves Pardubice 09:35; T1 reaches Pardubice 09:30. Chain is fine,
    # but only 5 min -- and this is the *only* reason it should fail.
    r = replay(
        [Leg("T1", "Praha", "Pardubice"), Leg("T4", "Pardubice", "Kolin")],
        w,
        min_transfer_sec=600,
    )
    assert not r and "need 600s" in r.reasons[0]


@case("all reasons collected, not just the first")
def _(w):
    r = replay([Leg("T99", "Praha", "Kolin"), Leg("T98", "Kolin", "Brno")], w)
    assert not r and len(r.reasons) == 2


# --- the bug the collapse fixed, guarded against regression -----------------


@case("transfer across former zone duplicates")
def _(w):
    # If the snapshot ever reverts to stop_id keying, T1->T3 at Kolin breaks.
    r = replay([Leg("T1", "Praha", "Kolin"), Leg("T3", "Kolin", "HavlBrod")], w)
    assert r, "station_id chaining regressed to stop_id"


def main():
    tmp = Path(tempfile.mkdtemp())
    try:
        w = build_world(tmp)
        failed = 0
        for name, fn in CASES:
            try:
                fn(w)
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