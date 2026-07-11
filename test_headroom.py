"""
Tests for headroom.py. Run: python test_headroom.py

The harness is tested with SCRIPTED models before any real one is called.
An oracle must score 100%, garbage must score 0, and a model that never emits
a parseable call must terminate rather than loop forever.

If the harness is wrong, the go/no-go number is about the harness.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from env import Env
from headroom import parse_call, run_episode, taxonomy, to_constraints
from replay import World

CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn

    return deco


def h(hh, mm=0):
    return hh * 3600 + mm * 60


ROWS = [
    ("T1", 1, "P", h(8), h(8)),
    ("T1", 2, "K", h(9), h(9, 2)),
    ("T2", 1, "K", h(9, 20), h(9, 20)),
    ("T2", 2, "HB", h(10, 20), h(10, 20)),
    # 120s after T1 arrives. The minimum is 180s, and exactly 180s is ALLOWED
    # (test_replay asserts that), so 09:03 would have been feasible.
    ("T3", 1, "K", h(9, 2), h(9, 2)),
    ("T3", 2, "HB", h(10), h(10)),
]
NAMES = {"P": "Praha", "K": "Kolín", "HB": "Havlíčkův Brod"}

INSTANCE = {
    "id": 0,
    "mode": "hub_leaf",
    "any_binds": False,
    "request": "Praha to Havlíčkův Brod, arriving by 11:00.",
    "origin": "P",
    "destination": "HB",
    "constraints": {
        "depart_after": 0,
        "arrive_before": h(11),
        "max_transfers": None,
        "avoid": [],
    },
}


def scripted(*replies):
    """A model that says these things in order, then repeats the last."""
    box = {"i": 0}

    def call(_messages):
        i = min(box["i"], len(replies) - 1)
        box["i"] += 1
        return replies[i]

    return call


def block(obj):
    return "reasoning here\n```json\n" + json.dumps(obj) + "\n```"


SUBMIT_OK = block(
    {
        "tool": "submit",
        "args": {
            "legs": [
                {"trip_id": "T1", "board": "P", "alight": "K"},
                {"trip_id": "T2", "board": "K", "alight": "HB"},
            ]
        },
    }
)
SUBMIT_TIGHT = block(
    {
        "tool": "submit",
        "args": {
            "legs": [
                {"trip_id": "T1", "board": "P", "alight": "K"},
                {"trip_id": "T3", "board": "K", "alight": "HB"},
            ]
        },
    }
)


# --- parsing ----------------------------------------------------------------


@case("parse: fenced json block")
def _(e):
    assert parse_call(block({"tool": "x", "args": {}}))["tool"] == "x"


@case("parse: the LAST block wins, not an example earlier in the text")
def _(e):
    text = block({"tool": "find_station", "args": {}}) + "\n" + block({"tool": "submit", "args": {}})
    assert parse_call(text)["tool"] == "submit"


@case("parse: bare json without a fence still parses")
def _(e):
    assert parse_call('I will call {"tool": "departures", "args": {}}')["tool"] == "departures"


@case("parse: prose with no call returns None")
def _(e):
    assert parse_call("I think we should take the train to Brno.") is None


@case("parse: args default to empty, never KeyError")
def _(e):
    assert parse_call('```json\n{"tool": "submit"}\n```')["args"] == {}


# --- episodes ---------------------------------------------------------------


@case("oracle: a correct submit scores solved")
def _(e):
    r = run_episode(e, INSTANCE, scripted(SUBMIT_OK), max_calls=10)
    assert r["solved"] and r["feasible"], r
    assert r["n_calls"] == 0  # submit is not a lookup
    assert taxonomy(r) == "solved"


@case("oracle: searching first still solves, and calls are counted")
def _(e):
    r = run_episode(
        e,
        INSTANCE,
        scripted(
            block({"tool": "find_station", "args": {"query": "Praha"}}),
            block({"tool": "departures", "args": {"station_id": "P", "after": "07:00"}}),
            SUBMIT_OK,
        ),
        max_calls=10,
    )
    assert r["solved"] and r["n_calls"] == 2, r


@case("impossible transfer is caught and named")
def _(e):
    r = run_episode(e, INSTANCE, scripted(SUBMIT_TIGHT), max_calls=10)
    assert not r["feasible"] and r["submitted"]
    assert taxonomy(r) == "infeasible: transfer too tight", (taxonomy(r), r["reasons"])


@case("a model that never submits burns the budget and stops")
def _(e):
    loop = block({"tool": "departures", "args": {"station_id": "P", "after": "07:00"}})
    r = run_episode(e, INSTANCE, scripted(loop), max_calls=4)
    assert not r["submitted"] and r["n_calls"] == 4
    assert taxonomy(r) == "never submitted: budget exhausted", taxonomy(r)
    assert r["reward"] < 0


@case("three CONSECUTIVE malformed replies end the episode")
def _(e):
    r = run_episode(e, INSTANCE, scripted("no json here"), max_calls=100)
    assert not r["submitted"]
    assert taxonomy(r) == "harness: unparseable replies"


@case("malformed count resets on recovery -- scattered mistakes do not kill")
def _(e):
    # A model that errs at turns 1, 3 and 5 but recovers each time must NOT be
    # executed. A cumulative counter killed 7 of 8 real rollouts this way.
    good = block({"tool": "departures", "args": {"station_id": "P", "after": "07:00"}})
    r = run_episode(
        e, INSTANCE, scripted("oops", good, "oops", good, "oops", SUBMIT_OK), max_calls=10
    )
    assert r["solved"], (taxonomy(r), r["malformed"])
    assert r["malformed"] == 3  # total, but never 3 in a row


@case("a submit truncated mid-json is recovered on the next turn")
def _(e):
    # max_tokens cuts the reply. The unterminated brace must not swallow the
    # rest of the text, and the episode must survive to try again.
    truncated = '```json\n{"tool": "submit", "args": {"legs": [{"trip_id": "T1"'
    r = run_episode(e, INSTANCE, scripted(truncated, SUBMIT_OK), max_calls=10)
    assert r["solved"] and r["malformed"] == 1


@case("parse: a fenced block that fails to parse must not suppress the fallback")
def _(e):
    # The fence matches, but the json inside is invalid (missing comma). With
    # `findall(...) or fallback`, the bare object below would never be seen.
    text = (
        '```json\n{"tool": "submit" "args": {}}\n```\n'
        'sorry, I meant: {"tool": "find_station", "args": {"query": "P"}}'
    )
    got = parse_call(text)
    assert got is not None and got["tool"] == "find_station", got


@case("parse: an unterminated brace does not swallow the real call")
def _(e):
    # No fence, and a stray `{` that never closes. A single-pass scanner
    # consumes the rest of the text and finds nothing.
    text = 'hmm {"maybe": [1, 2\nactually: {"tool": "submit", "args": {}}'
    got = parse_call(text)
    assert got is not None and got["tool"] == "submit", got


@case("a malformed reply is recoverable, not fatal")
def _(e):
    r = run_episode(e, INSTANCE, scripted("oops", SUBMIT_OK), max_calls=10)
    assert r["solved"] and r["malformed"] == 1


@case("bad tool name is an observation, the episode continues")
def _(e):
    r = run_episode(
        e, INSTANCE, scripted(block({"tool": "plan", "args": {}}), SUBMIT_OK), max_calls=10
    )
    assert r["solved"]
    assert "error" in r["trace"][0]["result"]


@case("submit with malformed legs scores as an empty submission")
def _(e):
    bad = block({"tool": "submit", "args": {"legs": [{"trip": "T1"}]}})
    r = run_episode(e, INSTANCE, scripted(bad), max_calls=10)
    assert not r["feasible"]


@case("model exceptions end the episode instead of crashing the run")
def _(e):
    def boom(_m):
        raise RuntimeError("rate limited")

    r = run_episode(e, INSTANCE, boom, max_calls=10)
    assert "error" in r and taxonomy(r) == "api error"


@case("violated constraints are reported by name, not just 'failed'")
def _(e):
    late = dict(INSTANCE)
    late["constraints"] = dict(INSTANCE["constraints"], arrive_before=h(10, 10))
    r = run_episode(e, late, scripted(SUBMIT_OK), max_calls=10)
    assert r["feasible"] and not r["solved"]
    assert taxonomy(r) == "violated: arrive_before", taxonomy(r)


@case("constraints round-trip from the instance json")
def _(e):
    c = to_constraints(INSTANCE)
    assert c.origin == "P" and c.arrive_before == h(11) and c.avoid == frozenset()


def main():
    tmp = Path(tempfile.mkdtemp())
    try:
        pd.DataFrame(
            ROWS, columns=["trip_id", "stop_sequence", "station_id", "arr", "dep"]
        ).to_parquet(tmp / "stop_times.parquet", index=False)
        ids = list(NAMES)
        pd.DataFrame(
            {
                "station_id": ids,
                "stop_name": [NAMES[i] for i in ids],
                "stop_lat": [50.0] * 3,
                "stop_lon": [14.0] * 3,
                "geolocatable": [True] * 3,
                "n_stop_ids": [1] * 3,
                "component": [0] * 3,
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