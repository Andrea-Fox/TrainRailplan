"""
Read an SFT trace in a human-friendly way.

    python show_trace.py sft_traces.jsonl              # summary + first multi-leg trace
    python show_trace.py sft_traces.jsonl --id 7       # a specific instance
    python show_trace.py sft_traces.jsonl --legs 3     # first trace with 3 legs
    python show_trace.py sft_traces.jsonl --dead-ends  # first trace containing a dead end
    python show_trace.py sft_traces.jsonl --list       # one line per trace

Each trace is a chat: the user gives the request and the tool results; the
assistant reasons and calls tools. This unfolds that back into readable turns,
and flags the two things the trace is meant to teach -- grounding (check before
submit) and recovery (a rejected leg followed by trying another).
"""

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

CALL_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)

# ANSI, degrade gracefully if piped
def c(code, s):
    return f"\033[{code}m{s}\033[0m" if sys.stdout.isatty() else s


def wrap(s, indent):
    return textwrap.fill(s, width=88, initial_indent=indent, subsequent_indent=indent)


def split_msg(content):
    """Assistant content -> (reasoning, call dict | None)."""
    m = CALL_RE.search(content)
    if not m:
        return content.strip(), None
    reasoning = content[: m.start()].strip()
    try:
        call = json.loads(m.group(1))
    except json.JSONDecodeError:
        call = None
    return reasoning, call


def render_call(call):
    if not call:
        return c("31", "[unparseable call]")
    args = ", ".join(f"{k}={v!r}" for k, v in call.get("args", {}).items())
    return c("36", f"{call['tool']}({args})")


def render_result(raw):
    try:
        r = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if "error" in r:
        return c("31", "ERROR: " + r["error"] + (f"  ({r['hint']})" if "hint" in r else ""))
    # compress the common shapes so the reasoning stays readable
    if "matches" in r:
        ms = ", ".join(f"{m['name']}={m['station_id']}" for m in r["matches"])
        extra = f"  (+{r['n_matches']-len(r['matches'])} more)" if r.get("truncated") else ""
        return f"{len(r['matches'])} match: {ms}{extra}"
    if "departures" in r:
        ds = " | ".join(f"{d['trip_id']} {d['departs']}->{d['towards']}" for d in r["departures"])
        return f"from {r['station']} after {r['after']}: {ds}" + ("  (truncated)" if r.get("truncated") else "")
    if "arrives" in r:
        via = f" via {', '.join(r['calls_at'])}" if r.get("calls_at") else ""
        return f"{r['trip_id']}: {r['from']} {r['departs']} -> {r['to']} {r['arrives']}{via}"
    return raw


def show(trace):
    msgs = trace["messages"]
    print(c("1", f"\n{'='*90}"))
    print(c("1", f"instance {trace['instance_id']}   legs={trace['n_legs']}   "
                 f"dead_end={trace['had_dead_end']}"))
    print(c("1", "="*90))

    step = 0
    for m in msgs:
        if m["role"] == "user":
            # first user message is the request; the rest are tool observations
            if step == 0:
                print(c("33", "\nREQUEST: ") + m["content"])
            else:
                print(wrap(c("90", "obs: ") + render_result(m["content"]), "       "))
        else:
            reasoning, call = split_msg(m["content"])
            step += 1
            print()
            print(wrap(c("37", reasoning), "  "))
            print("     -> " + render_call(call))
    print()


def summary(trace):
    msgs = trace["messages"]
    calls = [split_msg(m["content"])[1] for m in msgs if m["role"] == "assistant"]
    calls = [c for c in calls if c]
    n_leg_checks = sum(1 for c in calls if c["tool"] == "leg")
    errs = sum(1 for m in msgs if m["role"] == "user" and '"error"' in m["content"])
    tools = [c["tool"] for c in calls]
    return (f"inst {trace['instance_id']:>3}  legs={trace['n_legs']}  "
            f"calls={len(calls):>2}  leg_checks={n_leg_checks}  errors={errs}  "
            f"dead_end={'Y' if trace['had_dead_end'] else '.'}  "
            f"[{'>'.join(t[:3] for t in tools)}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--id", type=int)
    ap.add_argument("--legs", type=int)
    ap.add_argument("--dead-ends", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    traces = [json.loads(l) for l in Path(args.file).open()]

    if args.list:
        for t in traces:
            print(summary(t))
        de = sum(t["had_dead_end"] for t in traces)
        print(f"\n{len(traces)} traces, {de} with a dead end ({de/len(traces):.0%})")
        by_legs = {}
        for t in traces:
            by_legs.setdefault(t["n_legs"], 0)
            by_legs[t["n_legs"]] += 1
        print("by leg count:", dict(sorted(by_legs.items())))
        return

    if args.id is not None:
        pick = [t for t in traces if t["instance_id"] == args.id]
    elif args.legs is not None:
        pick = [t for t in traces if t["n_legs"] == args.legs]
    elif args.dead_ends:
        pick = [t for t in traces if t["had_dead_end"]]
    else:
        pick = [t for t in traces if t["n_legs"] >= 2] or traces

    if not pick:
        print("no trace matches that filter")
        return
    show(pick[0])


if __name__ == "__main__":
    main()
