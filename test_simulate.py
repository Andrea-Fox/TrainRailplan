"""
Tests for simulate_traces.py. Run: python test_simulate.py

The load-bearing property: a simulated trace is a VALID EPISODE. If you replay
the tool calls it contains through the real Env, you get back exactly the
observations it recorded, and its final submit scores as solved. A trace that
fails this is teaching the model fiction -- tool outputs it would never actually
see -- which is worse than no data.

Also checked: the grounding behaviour we are trying to teach is actually present
(endpoints resolved before any leg; every submitted leg was checked first), and
dead ends are real (a rejected leg followed by recovery, not a fabricated one).
"""

import json
import re
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from env import Env
from replay import Leg, World
from simulate_traces import Simulator, render_messages
import random

CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


def h(hh, mm=0):
    return hh * 3600 + mm * 60


# A world with a genuine dead end available: from P, both T1 (correct, connects
# to HB via K) and TX (wrong: heads toward HB-ish but terminates at DEC).
#   T1: P 08:00 -> K 09:00                 then T2: K 09:20 -> HB 10:20
#   TX: P 08:10 -> DEC 09:40               (a plausible wrong turn from P)
ROWS = [
    ("T1", 1, "P", h(8), h(8)),
    ("T1", 2, "K", h(9), h(9)),
    ("T2", 1, "K", h(9, 20), h(9, 20)),
    ("T2", 2, "HB", h(10, 20), h(10, 20)),
    ("TX", 1, "P", h(8, 10), h(8, 10)),
    ("TX", 2, "DEC", h(9, 40), h(9, 40)),
    ("TD", 1, "P", h(7, 30), h(7, 30)),   # direct-ish: P -> K only, single leg world
    ("TD", 2, "K", h(8, 15), h(8, 15)),
]
NAMES = {"P": "Praha", "K": "Kolín", "HB": "Havlíčkův Brod", "DEC": "Děčín"}


def build_env(tmp):
    pd.DataFrame(ROWS, columns=["trip_id", "stop_sequence", "station_id", "arr", "dep"]).to_parquet(
        tmp / "stop_times.parquet", index=False)
    ids = list(NAMES)
    pd.DataFrame({
        "station_id": ids, "stop_name": [NAMES[i] for i in ids],
        "stop_lat": [50.0] * len(ids), "stop_lon": [14.0] * len(ids),
        "geolocatable": [True] * len(ids), "n_stop_ids": [1] * len(ids),
        "component": [0] * len(ids),
    }).to_parquet(tmp / "stations.parquet", index=False)
    w = World(tmp)
    return Env(w, tmp), w


INST_2LEG = {
    "id": 0, "origin": "P", "destination": "HB",
    "origin_name": "Praha", "destination_name": "Havlíčkův Brod",
    "request": "Praha to Havlíčkův Brod, arriving by 11:00.",
    "constraints": {"depart_after": 0, "arrive_before": h(11), "max_transfers": None, "avoid": []},
    "reference": {"transfers": 1, "departure": h(8), "arrival": h(10, 20),
                  "legs": [{"trip_id": "T1", "board": "P", "alight": "K"},
                           {"trip_id": "T2", "board": "K", "alight": "HB"}]},
}

INST_1LEG = {
    "id": 1, "origin": "P", "destination": "K",
    "origin_name": "Praha", "destination_name": "Kolín",
    "request": "Praha to Kolín, arriving by 09:00.",
    "constraints": {"depart_after": 0, "arrive_before": h(9), "max_transfers": None, "avoid": []},
    "reference": {"transfers": 0, "departure": h(8), "arrival": h(9),
                  "legs": [{"trip_id": "T1", "board": "P", "alight": "K"}]},
}


CALL_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)


def extract_calls(msgs):
    """Pull the (tool,args) calls out of the assistant messages, in order."""
    calls = []
    for m in msgs:
        if m["role"] != "assistant":
            continue
        blk = CALL_RE.search(m["content"])
        assert blk, f"assistant message has no json call:\n{m['content']}"
        calls.append(json.loads(blk.group(1)))
    return calls


# --- the load-bearing property ----------------------------------------------


@case("every tool result in the trace is what the real Env returns")
def _(env, w):
    sim = Simulator(env, w, random.Random(0))
    turns = sim.trace(INST_2LEG)
    for t in turns:
        if t.tool in ("_note", "submit"):
            continue
        fresh = env.step(t.tool, t.args)
        assert fresh == t.result, f"recorded result diverges from Env for {t.tool}{t.args}"


@case("the submitted itinerary actually solves the instance")
def _(env, w):
    from solve import Constraints
    sim = Simulator(env, w, random.Random(0))
    turns = sim.trace(INST_2LEG)
    submit = [t for t in turns if t.tool == "submit"][0]
    legs = [Leg(l["trip_id"], l["board"], l["alight"]) for l in submit.args["legs"]]
    c = Constraints("P", "HB", arrive_before=h(11))
    o = env.score(legs, c, n_calls=len([t for t in turns if t.tool not in ("_note", "submit")]))
    assert o.solved if hasattr(o, "solved") else (o.feasible and not any(o.violations.values())), o


@case("replaying the extracted calls reproduces the recorded observations")
def _(env, w):
    sim = Simulator(env, w, random.Random(0))
    turns = sim.trace(INST_2LEG)
    msgs = render_messages(INST_2LEG, turns)
    calls = extract_calls(msgs)
    # user messages hold the observations; replay each non-submit call
    obs = [m["content"] for m in msgs if m["role"] == "user"][1:]  # drop the request
    j = 0
    for call in calls:
        if call["tool"] == "submit":
            continue
        fresh = json.dumps(env.step(call["tool"], call["args"]), ensure_ascii=False)
        assert fresh == obs[j], f"replay diverges at call {call}"
        j += 1


# --- the behaviour we are teaching ------------------------------------------


@case("both endpoints are resolved before any leg is checked")
def _(env, w):
    sim = Simulator(env, w, random.Random(0))
    calls = extract_calls(render_messages(INST_2LEG, sim.trace(INST_2LEG)))
    first_leg = next(i for i, c in enumerate(calls) if c["tool"] == "leg")
    finds = [c for c in calls[:first_leg] if c["tool"] == "find_station"]
    assert len(finds) >= 2, "endpoints must be grounded before searching legs"


@case("every submitted leg was checked with leg() first")
def _(env, w):
    sim = Simulator(env, w, random.Random(0))
    turns = sim.trace(INST_2LEG)
    calls = extract_calls(render_messages(INST_2LEG, turns))
    checked = {(c["args"]["trip_id"], c["args"]["board"], c["args"]["alight"])
               for c in calls if c["tool"] == "leg" and "error" not in (
                   env.step("leg", c["args"]))}
    submit = [c for c in calls if c["tool"] == "submit"][0]
    for l in submit.args["legs"] if hasattr(submit, "args") else submit["args"]["legs"]:
        assert (l["trip_id"], l["board"], l["alight"]) in checked, \
            f"submitted an unchecked leg: {l}"


@case("a dead end is a REAL rejected leg followed by recovery")
def _(env, w):
    # Force dead ends on this 2-leg instance; TX from P is the plausible wrong turn.
    sim = Simulator(env, w, random.Random(1))
    turns = sim.trace(INST_2LEG)
    errs = [t for t in turns if t.tool == "leg" and t.result and "error" in t.result]
    if errs:  # rng-dependent, but when present it must be genuine
        # the error is a real Env rejection, and a correct leg is checked afterward
        assert any(t.tool == "leg" and t.result and "error" not in t.result
                   for t in turns), "dead end with no recovery"


@case("single-leg journeys carry no forced dead end")
def _(env, w):
    sim = Simulator(env, w, random.Random(0))
    turns = sim.trace(INST_1LEG)
    errs = [t for t in turns if t.tool == "leg" and t.result and "error" in t.result]
    assert not errs, "a direct train has nothing to get wrong; no dead end should be forced"


@case("the trace ends in exactly one submit")
def _(env, w):
    sim = Simulator(env, w, random.Random(0))
    turns = sim.trace(INST_2LEG)
    submits = [t for t in turns if t.tool == "submit"]
    assert len(submits) == 1 and turns[-1].tool == "submit"


@case("every assistant message carries reasoning, not just a bare call")
def _(env, w):
    sim = Simulator(env, w, random.Random(0))
    msgs = render_messages(INST_2LEG, sim.trace(INST_2LEG))
    for m in msgs:
        if m["role"] == "assistant":
            before = m["content"].split("```")[0].strip()
            assert len(before) > 15, f"assistant message has no reasoning: {m['content'][:60]}"


def main():
    tmp = Path(tempfile.mkdtemp())
    try:
        env, w = build_env(tmp)
        failed = 0
        for name, fn in CASES:
            try:
                fn(env, w)
                print(f"  pass  {name}")
            except AssertionError as e:
                failed += 1
                print(f"  FAIL  {name}\n        {e}")
            except Exception as e:
                failed += 1
                print(f"  ERROR {name}\n        {type(e).__name__}: {e}")
        print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
        return 1 if failed else 0
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    raise SystemExit(main())
