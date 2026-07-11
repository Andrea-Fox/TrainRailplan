"""
Preflight for a local model server. Run this BEFORE headroom.py --backend openai.

    python preflight_qwen.py --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-3B-Instruct

Last time, 40 episodes came back as api errors because the client raced vLLM's
weight download and every request hit a server that was not up. This separates
"is the server answering" from "can the model do the task" so that never gets
silently folded into a solve rate of zero again.

Checks, in order:
  1. server reachable and /models lists the model
  2. a trivial completion returns
  3. the model can emit ONE well-formed tool call for our format
"""

import argparse
import json
import sys
import time

import requests


def wait_for_server(base_url: str, model: str, timeout_s: int = 600):
    url = base_url.rstrip("/") + "/models"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(url, timeout=5)
            if r.ok:
                ids = [m["id"] for m in r.json().get("data", [])]
                if model in ids:
                    print(f"server up, model loaded ({int(time.time()-t0)}s)")
                    return True
                print(f"server up but model not in {ids}")
                return False
        except requests.RequestException:
            pass
        print(f"  waiting for server... {int(time.time()-t0)}s", end="\r")
        time.sleep(3)
    print(f"\nserver not ready after {timeout_s}s")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    args = ap.parse_args()

    from openai import OpenAI

    if not wait_for_server(args.base_url, args.model):
        sys.exit(1)

    client = OpenAI(base_url=args.base_url, api_key="x")

    # 2. trivial completion
    r = client.chat.completions.create(
        model=args.model,
        max_tokens=16,
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
    )
    print("trivial completion:", repr(r.choices[0].message.content.strip()[:40]))

    # 3. can it emit one well-formed tool call?
    from headroom import SYSTEM, parse_call

    r = client.chat.completions.create(
        model=args.model,
        max_tokens=512,
        temperature=1.0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "I need to get from Kolín to Praha, arriving by 12:00."},
        ],
    )
    text = r.choices[0].message.content
    call = parse_call(text)
    print("\nfirst reply (truncated):")
    print("   ", text[:300].replace("\n", "\n    "))
    print("\nparsed tool call:", call)

    if call is None:
        print("\n>> Model did NOT emit a parseable call on turn 1.")
        print(">> This is the SFT signal: it cannot follow the tool format zero-shot.")
    elif call["tool"] not in {"find_station", "departures", "leg", "submit"}:
        print(f"\n>> Emitted an unknown tool {call['tool']!r} -- format partially understood.")
    else:
        print("\n>> Well-formed call. The model can at least start the task.")


if __name__ == "__main__":
    main()