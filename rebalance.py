"""
Rebalance a trace set to a target transfer-depth distribution.

    python rebalance.py sft_traces.jsonl --out sft_balanced.jsonl

Qwen cannot yet solve a direct train (instance 2). SFT data weighted 77% toward
2-3 transfer journeys teaches the hard cases while starving the easy ones the
model must master first. This subsamples to a shallow-weighted target so the
model learns to stand before it runs.

Works on the trace file (keys: n_legs) or, with --key transfers, on an instance
file. Subsamples only -- never invents -- so the output is a strict subset.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

# target fraction by leg count (legs = transfers + 1)
TARGET = {1: 0.15, 2: 0.35, 3: 0.35, 4: 0.15}  # 0,1,2,3 transfers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("--out", default="sft_balanced.jsonl")
    ap.add_argument("--key", default="n_legs", help="field holding leg count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max", type=int, default=None, help="cap total kept")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in Path(args.infile).open()]

    def legs(r):
        v = r.get(args.key)
        return v if args.key == "n_legs" else v + 1  # transfers -> legs

    buckets = {}
    for r in rows:
        buckets.setdefault(min(legs(r), 4), []).append(r)

    print("available:", {k: len(v) for k, v in sorted(buckets.items())})

    # Find the largest total N such that every bucket can supply its target
    # share. Limited by the scarcest bucket relative to its target weight.
    feasible_n = min(
        int(len(buckets.get(k, [])) / w) for k, w in TARGET.items() if w > 0
    )
    if args.max:
        feasible_n = min(feasible_n, args.max)

    kept = []
    for k, w in TARGET.items():
        want = round(feasible_n * w)
        pool = buckets.get(k, [])
        take = rng.sample(pool, min(want, len(pool)))
        kept.extend(take)
    rng.shuffle(kept)

    with Path(args.out).open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    got = Counter(min(legs(r), 4) for r in kept)
    print(f"\nkept {len(kept)}")
    print("distribution:")
    for k in sorted(got):
        print(f"  {k-1} transfers: {got[k]:>4}  ({got[k]/len(kept):.0%})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
