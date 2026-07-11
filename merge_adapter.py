"""
merge_adapter.py -- fold a trained LoRA adapter into the base weights so vLLM
can serve it for the re-probe.

    python merge_adapter.py --base Qwen/Qwen2.5-3B-Instruct \
        --adapter qwen-railplan-sft --out qwen-railplan-sft-merged

A LoRA run produces an adapter (a few MB of low-rank deltas), not a standalone
model. Two ways to serve it:

  (a) merge into the base -> a full model vLLM loads normally (this script), or
  (b) serve the base with vLLM --enable-lora and point at the adapter dir.

Merging is simpler to reason about and removes any adapter-loading quirks during
eval, at the cost of writing a full-size model to disk. Load in bf16 (NOT 4-bit)
for merging -- merging into a quantized base degrades the weights.
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="qwen-railplan-sft")
    ap.add_argument("--out", default="qwen-railplan-sft-merged")
    args = ap.parse_args()

    print(f"loading base {args.base} in bf16 ...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    print(f"attaching adapter {args.adapter} ...")
    model = PeftModel.from_pretrained(base, args.adapter)

    print("merging ...")
    model = model.merge_and_unload()

    print(f"saving to {args.out} ...")
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)

    print(f"\ndone. serve with:")
    print(f"  VLLM_ATTENTION_BACKEND=FLASH_ATTN python -m vllm.entrypoints.openai.api_server \\")
    print(f"      --model {args.out} --max-model-len 8192")
    print("then re-probe:")
    print(f"  python headroom.py data/cz_rail_20260714 instances.jsonl \\")
    print(f"      --n 10 --k 4 --backend openai --model {args.out} --out qwen_sft.jsonl")


if __name__ == "__main__":
    main()
