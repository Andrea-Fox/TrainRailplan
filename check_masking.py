"""
check_masking.py -- run FIRST on the GPU box, before train_sft.py.

    python check_masking.py --base Qwen/Qwen2.5-3B-Instruct --traces sft_traces.jsonl

The single most likely silent failure in SFT is loss masking not working. TRL's
assistant_only_loss relies on the tokenizer's chat template wrapping assistant
content in {% generation %} ... {% endgeneration %} markers. If Qwen's template
lacks them, TRL cannot mask, it trains on EVERYTHING including tool results, and
you get a model that hallucinates tool outputs -- the exact failure you are
trying to fix -- after burning two hours and a rented GPU.

This confirms, in 30 seconds and for free:
  1. the template has generation markers
  2. applying it to a real trace masks a clear majority of tokens
  3. the UNmasked tokens are the assistant's (reasoning + calls), and the
     tool results / request are masked
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--traces", default="sft_traces.jsonl")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    template = tok.chat_template or ""

    print("=" * 64)
    has_markers = "generation" in template
    print(f"chat template has {{% generation %}} markers: {has_markers}")
    if not has_markers:
        print(
            "\n>> The template does NOT mark assistant spans. assistant_only_loss\n"
            ">> will silently fail and train on tool results. FIX before training:\n"
            ">>   - use a tokenizer/template that includes {% generation %}, or\n"
            ">>   - supply a custom chat_template that wraps assistant content, or\n"
            ">>   - use TRL's DataCollatorForCompletionOnlyLM with an explicit\n"
            ">>     response template instead of assistant_only_loss.\n"
        )

    # Load one real multi-turn trace.
    row = json.loads(next(iter(Path(args.traces).open())))
    messages = row["messages"]

    # Render with the return_assistant_tokens_mask path TRL uses.
    try:
        enc = tok.apply_chat_template(
            messages,
            tokenize=True,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
    except Exception as e:
        print(f"\n>> apply_chat_template(return_assistant_tokens_mask=True) failed: {e}")
        print(">> This path is required for assistant_only_loss. Investigate the template.")
        return

    ids = enc["input_ids"]
    mask = enc.get("assistant_masks")
    if mask is None:
        print("\n>> No assistant_masks returned. Masking will not work.")
        return

    n = len(ids)
    trained = sum(mask)  # 1 = assistant token, contributes to loss
    print(f"\ntokens total:            {n}")
    print(f"assistant (trained):     {trained}  ({trained/n:.0%})")
    print(f"masked (request+results):{n - trained}  ({(n-trained)/n:.0%})")

    # Show a few trained vs masked spans so you can eyeball correctness.
    print("\nfirst 40 tokens, [T]=trained / [.]=masked:")
    toks = tok.convert_ids_to_tokens(ids[:40])
    line = " ".join(f"{'T' if m else '.'}{t}" for t, m in zip(toks, mask[:40]))
    print("  " + line.replace("Ġ", "_"))

    ok = has_markers and 0 < trained < n and trained / n < 0.6
    print("\n" + "=" * 64)
    if ok:
        print("OK. Assistant-only masking works. Safe to run train_sft.py.")
    else:
        print("NOT SAFE. Fix masking before training -- see notes above.")


if __name__ == "__main__":
    main()
