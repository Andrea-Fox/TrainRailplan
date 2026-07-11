"""
train_sft.py -- LoRA SFT of Qwen2.5-3B on grounded search traces.

    python train_sft.py sft_traces.jsonl --out qwen-railplan-sft

Teaches the procedure Qwen lacks: resolve endpoints, CHECK each leg before
trusting it, recover from a rejected leg, submit only confirmed legs. The traces
already demonstrate this; SFT's only job is to make the model imitate it.

THE ONE THING THAT MUST BE RIGHT: loss is computed on ASSISTANT tokens only.
The user turns are the request and the tool RESULTS -- environment outputs the
model must never learn to produce. Training on them teaches the model to
hallucinate tool responses, which is precisely the failure (fabricated
itineraries) we are here to fix. `assistant_only_loss=True` on the chat
template, plus a data collator that masks everything else, enforces this.

Runs on a single 24GB GPU (rented; will not fit on a laptop). Setup:

    pip install "transformers>=4.44" "trl>=0.11" peft accelerate bitsandbytes datasets

Qwen2.5-3B-Instruct is the base (NOT the AWQ quant -- you cannot LoRA-train a
pre-quantized model cleanly; load in bf16 or 4-bit nf4 and attach adapters).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_traces(path: str) -> Dataset:
    """Each row is {"messages": [...]} in OpenAI chat format. TRL applies the
    tokenizer's chat template and, with assistant_only_loss, masks non-assistant
    tokens. We keep only the messages field."""
    rows = []
    for line in Path(path).open():
        r = json.loads(line)
        rows.append({"messages": r["messages"]})
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces")
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--out", default="qwen-railplan-sft")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)  # effective batch 16
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--4bit", dest="four_bit", action="store_true",
                    help="load base in nf4 (QLoRA) to fit tighter VRAM")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # --- data: train/eval split -------------------------------------------
    ds = load_traces(args.traces).shuffle(seed=args.seed)
    n_eval = max(1, int(len(ds) * args.eval_frac))
    eval_ds = ds.select(range(n_eval))
    train_ds = ds.select(range(n_eval, len(ds)))
    print(f"train {len(train_ds)}  eval {len(eval_ds)}")

    # --- tokenizer ---------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- base model --------------------------------------------------------
    model_kwargs: dict = {"torch_dtype": torch.bfloat16}
    if args.four_bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.base, attn_implementation="flash_attention_2", **model_kwargs
    )

    # --- LoRA --------------------------------------------------------------
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Qwen2 attention + MLP projections. Covering the MLP matters for
        # learning a new behaviour rather than just re-weighting attention.
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # --- SFT config --------------------------------------------------------
    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=args.max_len,
        # THE CRITICAL FLAG. Loss on assistant tokens only -- never on the
        # request or the tool results.
        assistant_only_loss=True,
        packing=False,  # packing + assistant-only masking interact badly; keep off
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        peft_config=peft_config,
    )

    # Sanity: confirm the mask actually zeroes the non-assistant tokens before
    # burning a training run. If labels are all != -100, masking is off and the
    # model would learn to predict tool results.
    sample = trainer.train_dataset[0]
    if "labels" in sample:
        masked = sum(1 for x in sample["labels"] if x == -100)
        total = len(sample["labels"])
        print(f"label mask check: {masked}/{total} tokens masked "
              f"({masked/total:.0%}). Expect a clear majority masked "
              f"(request + tool results).")
        assert masked > 0, "NO tokens masked -- assistant_only_loss is not working"

    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"\nsaved adapter to {args.out}")
    print("re-probe with:")
    print(f"  # serve the merged/adapter model, then:")
    print(f"  python headroom.py data/cz_rail_20260714 instances.jsonl \\")
    print(f"      --n 10 --k 4 --backend openai --model {args.out} --out qwen_sft.jsonl")
    print("  # success gate: does it solve instance 2 (the direct train)?")


if __name__ == "__main__":
    main()
