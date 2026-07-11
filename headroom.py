"""
The headroom check. THE GO/NO-GO.

    python headroom.py data/cz_rail_20260714 instances.jsonl \\
        --n 30 --k 4 --temperature 1.0 --model claude-sonnet-5

Runs a strong prompted model over the environment with NO training, k times per
instance. The model is a MEASURING INSTRUMENT for the task, not a baseline for
the policy: its failures cannot be blamed on capability, so what they measure is
the environment.

THE MEAN SOLVE RATE IS NOT THE QUANTITY OF INTEREST.

GRPO's advantage is group-relative: A_i = r_i - mean(r_1..r_k). If all k rollouts
on an instance succeed, the advantage is zero. If all k fail, the advantage is
zero. Learning signal comes from DISAGREEMENT. A dataset of 50% trivial and 50%
impossible instances has a 50% solve rate and no gradient whatsoever.

So the go/no-go is the histogram of per-instance solve counts:

    mass at 0 and k     no gradient, however healthy the mean looks
    mass in between     the instances GRPO will actually learn from

An instance at 0/k with a strong model is as likely to be BROKEN as hard -- an
avoid constraint the tools cannot express looks exactly like a difficult one.
Inspect a few by hand before concluding anything about difficulty.

SECOND OUTPUT, free: the trajectories that end in a correct submit ARE the SFT
set. They contain real dead ends, because the model really hit them. Tracing the
reference solution instead would produce traces that walk straight to the answer,
teaching the SHAPE of a solution rather than the PROCESS of finding one -- and
GRPO would have nothing to bootstrap from.

PRE-REGISTERED PREDICTIONS (written before looking):
  1. Informative fraction lands above 20%.
  2. If the solve rate is low, failure mass sits in under-specified find_station
     queries, not in the tool hiding stations. Requests carry verbatim feed
     names, so an exact-match query always resolves.
  3. hub_hub solves better than leaf_leaf.
  4. any_binds=True solves worse than any_binds=False.
If (3) or (4) come out backwards, suspect the generator, not the model.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from env import MAX_CALLS, TOOLS_SPEC, Env
from replay import Leg, World
from solve import Constraints

SYSTEM = f"""You plan train journeys on a frozen Czech rail timetable.

You cannot see the timetable. You must discover it with tools.

{TOOLS_SPEC}

Respond with reasoning, then EXACTLY ONE tool call in a fenced json block:

```json
{{"tool": "find_station", "args": {{"query": "Kolín"}}}}
```

To finish:

```json
{{"tool": "submit", "args": {{"legs": [{{"trip_id": "143070", "board": "S9402.P1", "alight": "S8859.P1"}}]}}}}
```

Rules:
- All times are HH:MM on one service day. Times may exceed 24:00 for trains
  running past midnight (25:30 is half past one in the morning).
- Legs must chain: you alight where you next board, with at least 3 minutes
  to change.
- `avoid` means the train must not even pass through that station.
- Submit as soon as you have a valid itinerary. Tool calls cost you.
"""

CALL_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)


def balanced_objects(text: str):
    """Every balanced {...} substring, outermost first.

    A regex cannot do this: `{"tool": "x", "args": {}}` has nested braces, and
    `[^{}]*` excludes exactly the object we want. Small models frequently drop
    the code fence, so this fallback is load-bearing, not decoration.

    An unterminated `{` -- prose like "use {station_id}", or a reply truncated
    by max_tokens -- must not swallow everything after it. On failure, rescan
    from the next brace.
    """
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        for j in range(i, n):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    yield text[i : j + 1]
                    i = j + 1
                    break
        else:  # never closed: this brace is not an object
            i += 1


def parse_call(text: str) -> dict | None:
    """Last tool call wins -- models often show an example before the real one.

    Both sources are always tried. `A or B` was wrong: a fenced block that
    fails to parse (truncated by max_tokens, say) would suppress the fallback
    entirely, and the episode would be scored as if the model said nothing.
    """
    blocks = CALL_RE.findall(text) + list(balanced_objects(text))
    for b in reversed(blocks):
        try:
            obj = json.loads(b)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "tool" in obj:
            obj.setdefault("args", {})
            return obj
    return None


def to_constraints(inst: dict) -> Constraints:
    c = inst["constraints"]
    return Constraints(
        origin=inst["origin"],
        destination=inst["destination"],
        depart_after=c["depart_after"],
        arrive_before=c["arrive_before"],
        max_transfers=c["max_transfers"],
        avoid=frozenset(c["avoid"]),
    )


# --- model backends ---------------------------------------------------------


def anthropic_backend(model: str, temperature: float, max_tokens: int = 3000):
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def call(messages: list[dict]) -> str:
        r = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM,
            messages=messages,
            temperature=temperature,
        )
        return "".join(b.text for b in r.content if b.type == "text")

    return call


def openai_backend(model: str, base_url: str, temperature: float, max_tokens: int = 3000):
    """Any OpenAI-compatible endpoint: vLLM, Ollama, TGI."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "x"))

    def call(messages: list[dict]) -> str:
        r = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "system", "content": SYSTEM}] + messages,
        )
        return r.choices[0].message.content

    return call


def transformers_backend(model: str, temperature: float, max_tokens: int = 3000):
    """In-process HF generation. No server, no vLLM -- uses the same torch +
    transformers that trained the model.

    The SFT model was trained with a template whose end-of-turn token was never
    in the loss mask, so it never learned to EMIT <|im_end|> to stop -- it runs
    straight on, hallucinating the tool results and the rest of the conversation
    in a single generation. eos_token_id cannot fix that: the model does not
    produce the token to stop on.

    So we stop STRUCTURALLY: halt as soon as one complete ```json ... ``` fence
    has closed. That is exactly one tool call, which is what a turn should be.
    The harness then runs it against the real Env and feeds the real result
    back, forcing the model to take turns instead of soliloquising.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model)
    lm = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
        device_map=dev,
    )
    lm.eval()
    if "generation %}" in (tok.chat_template or ""):
        tok.chat_template = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct"
        ).chat_template

    do_sample = temperature > 0

    class StopOnClosedFence(StoppingCriteria):
        """Stop once the generated text contains a closing ``` that follows an
        opening ```json -- i.e. one complete tool-call block."""

        def __init__(self, prompt_len: int):
            self.prompt_len = prompt_len

        def __call__(self, input_ids, scores, **kw) -> bool:
            gen = tok.decode(input_ids[0][self.prompt_len:], skip_special_tokens=True)
            if "```" not in gen:
                return False
            # opened a json fence and closed a fence after it
            open_i = gen.find("```json")
            if open_i == -1:
                open_i = gen.find("```")
            close_i = gen.find("```", open_i + 3)
            return close_i != -1

    def call(messages: list[dict]) -> str:
        chat = [{"role": "system", "content": SYSTEM}] + messages
        prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(lm.device)
        plen = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = lm.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=0.95 if do_sample else None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                stopping_criteria=StoppingCriteriaList([StopOnClosedFence(plen)]),
            )
        text = tok.decode(out[0][plen:], skip_special_tokens=True)
        # Trim to the first complete fence so nothing past the first call leaks.
        open_i = text.find("```json")
        if open_i == -1:
            open_i = text.find("```")
        if open_i != -1:
            close_i = text.find("```", open_i + 3)
            if close_i != -1:
                text = text[: close_i + 3]
        return text

    return call



# --- episode ----------------------------------------------------------------


def run_episode(env: Env, inst: dict, call_model, max_calls: int) -> dict:
    c = to_constraints(inst)
    env.set_destination(inst["destination"])   # enable the km-to-dest signal
    messages = [{"role": "user", "content": inst["request"]}]
    trace, n_calls = [], 0
    legs = None
    malformed = 0  # CONSECUTIVE. Cumulative would execute a model that
    malformed_total = 0  # recovered from every mistake it made.
    error = None

    while n_calls < max_calls:
        try:
            text = call_model(messages)
        except Exception as e:  # rate limit, timeout: end honestly, keep the shape
            error = f"{type(e).__name__}: {e}"
            break

        messages.append({"role": "assistant", "content": text})
        call = parse_call(text)

        if call is None:
            malformed += 1
            malformed_total += 1
            if malformed >= 3:
                error = "3 consecutive unparseable replies"
                break
            # Say what went wrong. A truncated json object is a different
            # mistake from prose, and the model can fix the first one.
            hint = (
                "Your reply was cut off mid-json. Reason briefly, then emit the "
                "call."
                if text.rstrip().endswith(("{", ",", ":", '"'))
                else "No tool call found. Emit exactly one fenced json block."
            )
            messages.append({"role": "user", "content": hint})
            continue

        malformed = 0  # recovered

        if call["tool"] == "submit":
            raw = call["args"].get("legs", [])
            try:
                legs = [Leg(l["trip_id"], l["board"], l["alight"]) for l in raw]
            except (TypeError, KeyError):
                legs = []
            trace.append({"tool": "submit", "args": call["args"]})
            break

        n_calls += 1
        result = env.step(call["tool"], call["args"])
        trace.append({"tool": call["tool"], "args": call["args"], "result": result})
        messages.append(
            {"role": "user", "content": json.dumps(result, ensure_ascii=False)}
        )

    outcome = env.score(legs, c, n_calls)
    return {
        "instance_id": inst["id"],
        "mode": inst["mode"],
        "any_binds": inst["any_binds"],
        "request": inst["request"],
        "error": error,
        "n_calls": n_calls,
        "budget_exhausted": n_calls >= max_calls and legs is None,
        "malformed": malformed_total,
        "reward": outcome.reward,
        "feasible": outcome.feasible,
        "submitted": outcome.submitted,
        "solved": bool(
            error is None and outcome.feasible and not any(outcome.violations.values())
        ),
        "violations": outcome.violations,
        "reasons": outcome.reasons,
        "trace": trace,
        "messages": messages,
    }


def taxonomy(r: dict) -> str:
    """Why did this episode fail? '30% solve rate' is not actionable.
    '30%, and half the failures never submitted' is."""
    if r.get("error") == "3 consecutive unparseable replies":
        return "harness: unparseable replies"
    if r.get("error"):
        return "api error"
    if r["solved"]:
        return "solved"
    if not r["submitted"]:
        if r.get("budget_exhausted"):
            return "never submitted: budget exhausted"
        return "never submitted: gave up"
    if not r["feasible"]:
        first = (r["reasons"] or ["unknown"])[0]
        for pat, label in [
            ("to change at", "infeasible: transfer too tight"),
            ("does not run", "infeasible: trip not running"),
            ("does not call", "infeasible: station not on trip"),
            ("not the direction", "infeasible: wrong direction"),
            ("alight at", "infeasible: legs do not chain"),
            ("empty", "submitted nothing"),
        ]:
            if pat in first:
                return label
        return "infeasible: other"
    bad = [k for k, v in r["violations"].items() if v]
    return "violated: " + ",".join(sorted(bad)) if bad else "solved"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("instances")
    ap.add_argument("--n", type=int, default=30, help="instances")
    ap.add_argument("--k", type=int, default=4, help="rollouts per instance")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--backend", choices=["anthropic", "openai", "transformers"], default="anthropic")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--max-calls", type=int, default=MAX_CALLS)
    ap.add_argument("--out", default="headroom.jsonl")
    args = ap.parse_args()

    if args.k > 1 and args.temperature == 0:
        sys.exit("k>1 at temperature 0 gives k identical rollouts. Raise --temperature.")

    snap = Path(args.snapshot)
    env = Env(World(snap), snap)
    rows = [json.loads(l) for l in Path(args.instances).open()][: args.n]

    if args.backend == "anthropic":
        call_model = anthropic_backend(args.model, args.temperature)
    elif args.backend == "transformers":
        call_model = transformers_backend(args.model, args.temperature)
    else:
        call_model = openai_backend(args.model, args.base_url, args.temperature)

    results = []
    with Path(args.out).open("w") as f:
        for i, inst in enumerate(rows, 1):
            solved_k = 0
            for j in range(args.k):
                r = run_episode(env, inst, call_model, args.max_calls)
                r["why"] = taxonomy(r)
                r["rollout"] = j
                solved_k += bool(r.get("solved"))
                results.append(r)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            bar = "#" * solved_k + "." * (args.k - solved_k)
            print(f"{i:>3}/{len(rows)}  [{bar}] {solved_k}/{args.k}  {inst['request'][:64]}")

    report(results, args.k)


def report(results: list[dict], k: int):
    df = pd.DataFrame(results)
    n_inst = df.instance_id.nunique()
    solved, feasible = df.solved.mean(), df.feasible.mean()

    print(f"\n{'=' * 66}")
    print(f"episodes     {len(df)}  ({n_inst} instances x {k} rollouts)")
    print(f"solve rate   {solved:.0%}   (pooled over episodes)")
    print(f"feasible     {feasible:.0%}")
    print(f"mean reward  {df.reward.mean():.3f}")
    print(f"tool calls   median {df.n_calls.median():.0f}  p90 {df.n_calls.quantile(.9):.0f}")

    print("\nwhy:")
    for why, c in Counter(df.why).most_common():
        print(f"  {c:>4}  {why}")

    # --- the actual go/no-go ------------------------------------------------
    #
    # GRPO's advantage is group-relative: A_i = r_i - mean(r_1..r_k). If every
    # rollout on an instance succeeds, the advantage is zero. If every rollout
    # fails, the advantage is zero. LEARNING SIGNAL COMES FROM DISAGREEMENT.
    #
    # A dataset of 50% trivial and 50% impossible has a 50% solve rate and no
    # gradient at all. The mean is not the quantity of interest.
    per = df.groupby("instance_id").solved.sum()
    hist = per.value_counts().reindex(range(k + 1), fill_value=0).sort_index()

    print(f"\n{'=' * 66}\nper-instance solve count (the GRPO gradient lives here)")
    width = max(1, hist.max())
    for s, c in hist.items():
        tag = "  <- no gradient" if s in (0, k) else ""
        print(f"  {s}/{k}  {c:>4}  {'#' * int(30 * c / width)}{tag}")

    informative = per.between(1, k - 1).mean()
    print(f"\ninformative instances (0 < solved < k): {informative:.0%}")
    if k > 1:
        # Variance of a Bernoulli(p) averaged over instances: what GRPO can use.
        p = per / k
        print(f"mean per-instance variance p(1-p):     {(p * (1 - p)).mean():.3f}"
              f"   (max possible {0.25:.3f})")

    print("\nsolve rate by mode:")
    print(df.groupby("mode").solved.agg(["mean", "size"]).to_string())
    print("\nsolve rate by whether a preference binds:")
    print(df.groupby("any_binds").solved.agg(["mean", "size"]).to_string())

    print(f"\n{'=' * 66}")
    if informative < 0.20:
        if solved > 0.8:
            print("NO HEADROOM: instances are trivial. Harden the sampler.")
        elif solved < 0.15:
            print("NO GRADIENT: instances are impossible. Suspect the interface,")
            print("not the policy. Inspect a 0/k instance BY HAND before blaming")
            print("difficulty -- an avoid that the tools cannot express looks")
            print("exactly like a hard instance.")
        else:
            print("BIMODAL: the mean looks healthy but instances are trivial or")
            print("impossible, never in between. GRPO has almost nothing to learn.")
    else:
        seed = int(df.solved.sum())
        print(f"HEADROOM. {informative:.0%} of instances give a gradient.")
        print(f"{seed} solved trajectories are your SFT seed.")


if __name__ == "__main__":
    main()
