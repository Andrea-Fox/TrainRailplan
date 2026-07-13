"""
simulate_traces.py -- SFT trajectory synthesizer.

    python simulate_traces.py data/cz_rail_20260714 instances.jsonl \
        --out sft_traces.jsonl --dead-end-rate 0.45

It does NOT render the reference solution as a straight walk to the answer.
That teaches the SHAPE of a solution, not the PROCESS of finding one: a model
so trained never sees a dead end, so a real rejected leg leaves it with nothing
to imitate, and it thrashes. Qwen's measured failure -- fabricating an itinerary
after 2-3 calls -- is exactly this: it never learned that a tool result must be
checked and that a wrong turn is recoverable.

So each trace demonstrates the behaviour Qwen lacks:

  * resolve both station names first (grounding the endpoints)
  * before trusting a train, CHECK it with leg()
  * sometimes the checked train is wrong -> read the error, back up, try another
  * only submit legs the tools actually confirmed

Every observation in a trace is REAL: the simulator drives the live Env, so the
tool outputs are exactly what the policy would see. It has privileged knowledge
of the reference answer, which it uses only to steer and to place instructive
mistakes -- never to fabricate a result.

Dead ends (chosen last night):
  (a) right direction, wrong train -- board a departure heading toward the
      destination that does not actually connect; check, fail, recover. This is
      the one that most directly teaches verify-before-trust.
  (b) plausible hub, no onward service in time -- rarer.

Rate is stratified by depth: more dead ends on multi-leg traces where real
search branches, none forced on single-leg direct trains where there is nothing
to get wrong.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from env import Env, hhmm
from headroom import to_constraints
from replay import Leg, World


@dataclass
class Turn:
    """One assistant step: reasoning text, then exactly one tool call. The
    observation is the real Env result, attached after the call is made."""

    think: str
    tool: str
    args: dict
    result: dict | None = None


# --- templated reasoning ----------------------------------------------------
# Terse enough to train fast, explicit enough that a small model can learn WHEN
# to call WHAT. Variety via a small pool so SFT does not overfit one phrasing.

def pick(rng, options):
    return rng.choice(options)


def think_resolve_origin(rng, name):
    return pick(rng, [
        f"I need to plan a journey to a fixed deadline. First, resolve the origin "
        f"'{name}' to a station id so later calls are unambiguous.",
        f"Start by grounding the endpoints. Look up the origin '{name}'.",
        f"Before searching for trains I need station ids. Resolve the origin '{name}' first.",
    ])


def think_resolve_dest(rng, name):
    return pick(rng, [
        f"Now resolve the destination '{name}' the same way.",
        f"Origin is grounded. Look up the destination '{name}'.",
        f"Next, get the id for the destination '{name}'.",
    ])


def think_list_departures(rng, station_name, after):
    return pick(rng, [
        f"I'm at {station_name}. List the trains leaving here after {after}; I'll "
        f"prefer one whose terminus is closer to the destination.",
        f"From {station_name}, see what departs after {after} and pick the one "
        f"heading most toward the goal.",
        f"Check departures from {station_name} after {after}, using distance-to-"
        f"destination to rank the candidates before committing.",
    ])


def think_check_leg(rng, trip, frm, to):
    # These explicitly contrast what departures TOLD us (the train leaves here)
    # with what only leg() can CONFIRM (it actually reaches my alight). The model
    # was skipping verification because the old reasoning was interchangeable
    # boilerplate; making each one contentful -- naming the specific gap between
    # "leaves here" and "stops where I need" -- both teaches the distinction and
    # resists being compressed away as filler.
    return pick(rng, [
        f"Departures only tells me {trip} leaves {frm} -- not that it stops at {to}. "
        f"I have to confirm that with leg() before it can go in the itinerary.",
        f"{trip} appears in the departures list, but a listing isn't a confirmed leg: "
        f"it might not call at {to} at all. Check {frm} -> {to} with leg().",
        f"Before I trust {trip} for {frm} -> {to}, verify it with leg(). Seeing it in "
        f"departures means it departs {frm}, not that it reaches {to}.",
        f"I won't submit {trip} on the strength of a departures listing -- that only "
        f"shows departures. leg() is the only thing that confirms it serves {to}.",
        f"Check {trip} actually runs {frm} -> {to} with leg(). Departures showed it "
        f"leaving {frm}; whether it stops at {to} is a separate fact I must verify.",
    ])


def think_dead_end_recover(rng, trip, reason):
    return pick(rng, [
        f"{trip} doesn't work: {reason}. That's fine -- back up and try another "
        f"departure rather than guess.",
        f"So {trip} is out ({reason}). Return to the departures and pick a different train.",
        f"{trip} fails ({reason}). Don't submit it. Reconsider the other options.",
    ])


def think_leg_ok(rng, trip, to, arr):
    return pick(rng, [
        f"Confirmed: {trip} reaches {to} at {arr}. Keep this leg.",
        f"Good, {trip} arrives {to} at {arr}. That leg checks out.",
        f"{trip} works, arriving {to} at {arr}. On to the next hop.",
    ])


def think_submit(rng, n):
    if n == 1:
        return pick(rng, [
            "That single leg gets me there within the deadline and I've verified it. Submit.",
            "One confirmed leg is the whole journey. Submit it.",
        ])
    return pick(rng, [
        f"All {n} legs are individually confirmed and they chain with time to change. Submit.",
        f"I've verified every one of the {n} legs and the transfers work. Submit the itinerary.",
    ])


def think_premature_submit(rng):
    # The agent is TEMPTED to submit straight from the departures listing without
    # confirming the last leg -- the exact mistake the trained model makes. It
    # does so, and the retryable submit rejects it. Modelled only as a prelude to
    # the correction below, never in isolation.
    return pick(rng, [
        "The departures listing shows a train that looks right for the last hop -- "
        "let me just submit the whole thing.",
        "This last trip appears in the departures, so I'll put it in and submit now.",
        "I'll assume the final train from the listing works and submit the itinerary.",
    ])


def think_submit_rejected_recover(rng, reason):
    # The rejection lands and the agent draws the right lesson: a departures
    # listing is not a confirmation; verify with leg() and resubmit.
    return pick(rng, [
        f"Rejected -- {reason}. That's the lesson: a departures listing isn't a "
        f"confirmed leg. I need to check it with leg() before submitting.",
        f"Infeasible ({reason}). I submitted a leg I only saw in departures, never "
        f"confirmed. Verify it properly with leg(), then resubmit.",
        f"The submission bounced: {reason}. Right -- I skipped confirming that leg. "
        f"leg() first, then submit what it confirms.",
    ])


# --- the simulator ----------------------------------------------------------


class Simulator:
    def __init__(self, env: Env, world: World, rng: random.Random):
        self.env = env
        self.w = world
        self.rng = rng

    def _towards_dest_wrong_trains(self, station_id, after_sec, dest_id, correct_trip):
        """Candidate dead ends of type (a): trains leaving `station_id` that head
        roughly toward the destination but are NOT the correct next leg. 'Toward'
        is judged by the real Env departures output (its `towards` field), so the
        mistake is one a reasonable searcher could actually make."""
        res = self.env.departures(station_id, hhmm(after_sec))
        cands = []
        for d in res.get("departures", []):
            if d["trip_id"] == correct_trip:
                continue
            # a plausible wrong turn: does this trip even reach the destination?
            leg = self.env.leg(d["trip_id"], station_id, dest_id)
            if "error" in leg:
                cands.append(d["trip_id"])  # heads off, doesn't connect: instructive
        return cands

    def trace(self, inst: dict) -> list[Turn] | None:
        rng = self.rng
        turns: list[Turn] = []
        o, d = inst["origin"], inst["destination"]
        ref_legs = [Leg(l["trip_id"], l["board"], l["alight"]) for l in inst["reference"]["legs"]]

        # Turn on the navigational distance signal for this episode. Every
        # departures()/leg() the simulator drives now carries km-to-destination,
        # so the recorded observations match exactly what the policy will see at
        # train and eval time -- and the reasoning below can refer to it.
        self.env.set_destination(d)

        # 1. resolve origin
        turns.append(self._call(think_resolve_origin(rng, inst["origin_name"]),
                                 "find_station", {"query": inst["origin_name"]}))
        # 2. resolve destination
        turns.append(self._call(think_resolve_dest(rng, inst["destination_name"]),
                                 "find_station", {"query": inst["destination_name"]}))

        # depart no earlier than the request's floor
        after = inst["constraints"]["depart_after"] or 0

        # 3..N: walk the reference legs, grounding each, with optional dead ends
        n_legs = len(ref_legs)
        dead_end_budget = self._dead_ends_for(n_legs)

        for i, leg in enumerate(ref_legs):
            board = leg.board
            # list departures from the current board station
            turns.append(self._call(
                think_list_departures(rng, self.env.names[board], hhmm(after)),
                "departures", {"station_id": board, "after": hhmm(after)},
            ))

            # optionally inject a type-(a) dead end before the correct leg
            if dead_end_budget > 0 and rng.random() < 0.7:
                wrong = self._towards_dest_wrong_trains(board, after, d, leg.trip_id)
                if wrong:
                    wt = rng.choice(wrong)
                    turns.append(self._call(
                        think_check_leg(rng, wt, self.env.names[board], self.env.names[d]),
                        "leg", {"trip_id": wt, "board": board, "alight": d},
                    ))
                    reason = self._reason(turns[-1].result)
                    turns.append(Turn(
                        think_dead_end_recover(rng, wt, reason),
                        "_note", {},  # not a tool call; a reasoning-only beat
                    ))
                    dead_end_budget -= 1

            # the correct leg: check it, confirm it
            turns.append(self._call(
                think_check_leg(rng, leg.trip_id, self.env.names[board], self.env.names[leg.alight]),
                "leg", {"trip_id": leg.trip_id, "board": board, "alight": leg.alight},
            ))
            r = turns[-1].result
            if "error" in r:
                # the reference itself failed to validate -- should never happen;
                # skip this instance rather than emit a broken trace.
                return None
            # Confirmation reasoning references the distance signal when present,
            # so the model learns to READ it: "arrived, and it's N km closer".
            km = r.get("to_km_to_dest")
            confirm = think_leg_ok(rng, leg.trip_id, self.env.names[leg.alight], r["arrives"])
            if km is not None:
                confirm += f" That leaves me {km} km from the destination."
            turns.append(Turn(confirm, "_note", {}))
            after = parse_arr(r["arrives"])  # next hop leaves after this arrival

        # B-pattern (~25% of multi-leg traces): before the correct submit, model
        # the exact mistake the trained policy makes -- grab a plausible train
        # from the departures listing WITHOUT confirming it and submit. Retryable
        # submit rejects it; the agent recovers by verifying and resubmitting. Now
        # that submit is retryable this is an executable pattern, and it teaches
        # that a departures listing is not a confirmation. Kept a minority so the
        # dominant signal stays clean verify-then-submit; the wrong submit never
        # appears in isolation, always immediately followed by the rejection and
        # the correction.
        do_B = n_legs >= 2 and rng.random() < 0.25
        if do_B:
            # take a wrong last leg the agent "saw in departures" but never checked
            last = ref_legs[-1]
            wrong_trains = self._towards_dest_wrong_trains(
                last.board, after, d, last.trip_id)
            if wrong_trains:
                wrong_tid = rng.choice(wrong_trains)
                premature = [{"trip_id": l.trip_id, "board": l.board, "alight": l.alight}
                             for l in ref_legs[:-1]]
                premature.append({"trip_id": wrong_tid, "board": last.board,
                                  "alight": last.alight})
                cand = [Leg(x["trip_id"], x["board"], x["alight"]) for x in premature]
                verdict = self.env.check_submit(cand, to_constraints(inst))
                if not verdict.get("feasible", False):   # only if it really rejects
                    turns.append(Turn(think_premature_submit(rng), "submit",
                                      {"legs": premature}, verdict))
                    reason = verdict.get("reasons", [verdict.get("error", "infeasible")])
                    reason = reason[0] if isinstance(reason, list) and reason else "infeasible"
                    turns.append(Turn(
                        think_submit_rejected_recover(rng, self._reason_from_submit(reason)),
                        "_note", {}))
                    # now confirm the correct last leg properly with leg()
                    turns.append(self._call(
                        think_check_leg(rng, last.trip_id, self.env.names[last.board],
                                        self.env.names[last.alight]),
                        "leg", {"trip_id": last.trip_id, "board": last.board,
                                "alight": last.alight},
                    ))
                    rr = turns[-1].result
                    if "error" not in rr:
                        turns.append(Turn(
                            think_leg_ok(rng, last.trip_id, self.env.names[last.alight],
                                         rr["arrives"]), "_note", {}))

        # final: submit the fully-confirmed itinerary
        turns.append(Turn(
            think_submit(rng, n_legs),
            "submit",
            {"legs": [{"trip_id": l.trip_id, "board": l.board, "alight": l.alight}
                      for l in ref_legs]},
        ))
        return turns

    def _dead_ends_for(self, n_legs: int) -> int:
        """Stratify: nothing forced on a direct train; up to n_legs-1 on longer
        journeys, gated by the global rate at call time."""
        if n_legs <= 1:
            return 0
        return n_legs - 1

    def _reason_from_submit(self, reason: str) -> str:
        """Phrase a check_submit rejection reason the way the recover-reasoning
        expects -- same vocabulary as leg-error reasons."""
        r = reason.lower()
        if "does not call" in r or "not call" in r:
            return "that train doesn't stop where I need"
        if "does not run" in r or "not run" in r:
            return "that train isn't running today"
        if "other way" in r or "opposite" in r:
            return "that train runs the wrong direction"
        if "chain" in r or "connect" in r:
            return "the legs don't connect in time"
        return reason

    def _reason(self, result: dict) -> str:
        if "error" in result:
            e = result["error"]
            if "other way" in e:
                return "it runs the opposite direction"
            if "does not call" in e:
                return "it doesn't stop where I need"
            if "does not run" in e:
                return "it isn't running today"
            return e
        return "it doesn't connect in time"

    def _call(self, think: str, tool: str, args: dict) -> Turn:
        result = self.env.step(tool, args)
        return Turn(think, tool, args, result)


def parse_arr(hhmm_str: str) -> int:
    h, m = hhmm_str.split(":")
    return int(h) * 3600 + int(m) * 60


# --- rendering to a chat trace ----------------------------------------------


def render_messages(inst: dict, turns: list[Turn]) -> list[dict]:
    """Flatten turns into an OpenAI-style message list: the request, then
    alternating assistant (reasoning + call) / user (tool result) messages.
    Reasoning-only beats (_note) fold into the NEXT assistant message so the
    trace stays a clean call/observation alternation."""
    msgs = [{"role": "user", "content": inst["request"]}]
    pending_think: list[str] = []

    for t in turns:
        if t.tool == "_note":
            pending_think.append(t.think)
            continue
        think = " ".join(pending_think + [t.think])
        pending_think = []
        call = {"tool": t.tool, "args": t.args}
        msgs.append({
            "role": "assistant",
            "content": f"{think}\n\n```json\n{json.dumps(call, ensure_ascii=False)}\n```",
        })
        if t.tool != "submit":
            msgs.append({
                "role": "user",
                "content": json.dumps(t.result, ensure_ascii=False),
            })
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("instances")
    ap.add_argument("--out", default="sft_traces.jsonl")
    ap.add_argument("--dead-end-rate", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    snap = Path(args.snapshot)
    world = World(snap)
    env = Env(world, snap)
    rng = random.Random(args.seed)
    sim = Simulator(env, world, rng)

    rows = [json.loads(l) for l in Path(args.instances).open()]
    kept, skipped, with_dead_end = 0, 0, 0

    with Path(args.out).open("w") as f:
        for inst in rows:
            # global rate gate: does THIS trace get to carry dead ends at all?
            allow_dead_ends = rng.random() < args.dead_end_rate
            saved_budget = sim._dead_ends_for
            if not allow_dead_ends:
                sim._dead_ends_for = lambda n: 0
            turns = sim.trace(inst)
            sim._dead_ends_for = saved_budget

            if turns is None:
                skipped += 1
                continue
            had_de = any(t.tool == "_note" and "back up" in t.think.lower()
                         or t.tool == "_note" and "out" in t.think.lower()
                         for t in turns)
            # count a dead end by presence of a recover-note before a confirm
            had_de = sum(1 for t in turns if t.tool == "leg"
                         and t.result and "error" in t.result) > 0
            with_dead_end += had_de

            msgs = render_messages(inst, turns)
            f.write(json.dumps({
                "instance_id": inst["id"],
                "n_legs": inst["reference"]["transfers"] + 1,
                "had_dead_end": had_de,
                "messages": msgs,
            }, ensure_ascii=False) + "\n")
            kept += 1

    print(f"kept {kept}, skipped {skipped} (reference failed to validate)")
    print(f"traces with >=1 dead end: {with_dead_end} ({with_dead_end/max(kept,1):.0%})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()