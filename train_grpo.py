"""
train_grpo.py -- GRPO for multi-turn tool-use, from scratch, sized for 24GB.

    python train_grpo.py data/cz_rail_20260714 instances.jsonl \
        --adapter qwen-railplan-sft --out qwen-railplan-grpo

WHY FROM SCRATCH. TRL's GRPOTrainer assumes a single prompt -> single completion
-> scalar reward. Our rollout is a whole EPISODE: generate a tool call, run the
real Env, feed the result back, repeat, submit, then score with env.score(). The
reward lands only at the end, and only the ASSISTANT tokens (the model's calls
and reasoning) should receive gradient -- never the tool results, which are the
environment's, not the policy's. None of that fits the single-turn abstraction,
so the loop is explicit here.

WHAT GRPO DOES. For each instance, sample k episodes. Their rewards form a group.
The advantage of episode i is A_i = (r_i - mean) / (std + eps) -- purely relative
to its own group, which is why no value network is needed. Then the policy is
nudged to raise the log-prob of the assistant tokens in above-average episodes
and lower them in below-average ones, with a KL penalty to the frozen reference
(the SFT model) so it cannot wander off and collapse. LEARNING SIGNAL IS THE
SPREAD WITHIN A GROUP: if all k episodes score the same, A_i = 0 for all of them
and that instance teaches nothing this step. (This is exactly the reward-spread
the probe measured before committing to GRPO.)

24GB FIT. Policy is the LoRA adapter only (tiny optimizer state). The reference
is the SAME base with the adapter disabled -- no second model in memory. The
base loads in 4-bit. Rollouts are generated in-process with transformers (no
vLLM). One episode's tokens at a time; gradient accumulates over the group.

RISK. This is the most intricate component and the least battle-tested. Validate
on a tiny run first (--n 2 --k 4 --steps 3) and watch that: (1) rewards vary
within groups, (2) loss is finite, (3) KL stays small and positive. Only then
scale up.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from env import Env, MAX_CALLS
from headroom import SYSTEM, parse_call, to_constraints
from replay import Leg, World


# --- one episode, with token-level bookkeeping ------------------------------


@dataclass
class Episode:
    input_ids: torch.Tensor      # (T,) full conversation: prompt + all turns
    policy_mask: torch.Tensor    # (T,) 1 where the POLICY generated the token
    reward: float
    n_calls: int
    solved: bool


IM_END = "<|im_end|>"


def build_prompt_ids(tok, messages: list[dict]) -> torch.Tensor:
    chat = [{"role": "system", "content": SYSTEM}] + messages
    text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").input_ids[0]


@torch.no_grad()
def rollout(model, tok, env: Env, inst: dict, max_calls: int,
            temperature: float, max_new: int, device) -> Episode:
    """Run one full episode. Returns the token sequence and a mask marking which
    tokens the policy produced (only those get a gradient)."""
    c = to_constraints(inst)
    messages = [{"role": "user", "content": inst["request"]}]

    # Grow two parallel sequences: the token ids, and a mask of policy tokens.
    ids = build_prompt_ids(tok, messages).to(device)
    mask = torch.zeros_like(ids)

    n_calls, malformed, legs, error = 0, 0, None, None
    im_end_id = tok.convert_tokens_to_ids(IM_END)

    while n_calls < max_calls:
        # generate one assistant turn, stopping at the closing fence
        gen = model.generate(
            input_ids=ids.unsqueeze(0),
            max_new_tokens=max_new,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.95 if temperature > 0 else None,
            pad_token_id=tok.pad_token_id or im_end_id,
            eos_token_id=im_end_id,
        )[0]
        new = gen[ids.shape[0]:]                     # freshly generated tokens
        text = tok.decode(new, skip_special_tokens=True)

        # trim decode + tokens to the first complete ```...``` fence
        cut_text, cut_len = trim_to_first_fence(tok, new, text)
        new = new[:cut_len]

        ids = torch.cat([ids, new])
        mask = torch.cat([mask, torch.ones_like(new)])   # policy tokens: gradient

        call = parse_call(cut_text)
        if call is None:
            malformed += 1
            if malformed >= 3:
                error = "unparseable"
                break
            obs = {"error": "no tool call found; emit one json block"}
        elif call["tool"] == "submit":
            raw = call["args"].get("legs", [])
            try:
                legs = [Leg(l["trip_id"], l["board"], l["alight"]) for l in raw]
            except (TypeError, KeyError):
                legs = []
            break
        else:
            malformed = 0
            n_calls += 1
            obs = env.step(call["tool"], call["args"])

        # append the tool result as a user turn -- NOT policy tokens (mask 0)
        obs_text = tok.apply_chat_template(
            [{"role": "user", "content": json.dumps(obs, ensure_ascii=False)}],
            tokenize=False, add_generation_prompt=True,
        )
        obs_ids = tok(obs_text, return_tensors="pt").input_ids[0].to(device)
        ids = torch.cat([ids, obs_ids])
        mask = torch.cat([mask, torch.zeros_like(obs_ids)])

        if ids.shape[0] > 3000:      # hard context guard: backward must fit 24GB
            error = "context overflow"
            break

    outcome = env.score(legs, c, n_calls)
    solved = bool(error is None and outcome.feasible
                  and not any(outcome.violations.values()))
    return Episode(ids.cpu(), mask.cpu(), outcome.reward, n_calls, solved)


def trim_to_first_fence(tok, new_ids: torch.Tensor, text: str):
    """Cut generated text+tokens at the end of the first ```...``` block, since
    the SFT model never learned to emit im_end and will run on otherwise."""
    open_i = text.find("```json")
    if open_i == -1:
        open_i = text.find("```")
    if open_i == -1:
        return text, new_ids.shape[0]
    close_i = text.find("```", open_i + 3)
    if close_i == -1:
        return text, new_ids.shape[0]
    cut_text = text[: close_i + 3]
    # find how many tokens correspond to cut_text by re-encoding prefix length
    for n in range(1, new_ids.shape[0] + 1):
        if tok.decode(new_ids[:n], skip_special_tokens=True).find("```", open_i + 3) != -1:
            return cut_text, n
    return cut_text, new_ids.shape[0]


# --- log-probs and the GRPO loss --------------------------------------------


def token_logprobs(model, ids: torch.Tensor, device) -> torch.Tensor:
    """Per-token log-prob of ids[1:] under model, shape (T-1,).

    Avoids materialising a (T-1, V) fp32 log-softmax -- with V=151k that tensor
    alone is gigabytes and OOMs the backward. Instead: logprob = logit_target -
    logsumexp(logits), computed per position in the model's own dtype. Only the
    target-token logit is gathered; the full distribution is never stored.
    """
    ids = ids.to(device).unsqueeze(0)
    logits = model(ids).logits[0, :-1]                 # (T-1, V), bf16
    targets = ids[0, 1:]                               # (T-1,)
    tgt_logit = logits.gather(1, targets.unsqueeze(1)).squeeze(1)  # (T-1,)
    lse = torch.logsumexp(logits, dim=-1)              # (T-1,)
    return tgt_logit - lse


def grpo_step(model, tok, episodes: list[Episode], kl_coef: float, device):
    """One optimisation step over one instance's group of episodes.

    Advantage is group-relative. The loss raises log-prob of assistant tokens in
    above-average episodes, lowered for below-average, with a per-token KL to the
    reference (adapter-disabled base). Only masked (policy) tokens contribute.
    """
    rewards = torch.tensor([e.reward for e in episodes])
    if rewards.std() < 1e-6:
        return None      # no spread in this group: nothing to learn this step
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

    total_loss, total_kl, total_tokens = 0.0, 0.0, 0
    for e, a in zip(episodes, adv):
        ids = e.input_ids
        m = e.policy_mask[1:].to(device).float()       # align with logp[1:]
        if m.sum() == 0:
            continue

        logp = token_logprobs(model, ids, device)      # current policy, grad on
        with torch.no_grad():
            with model.disable_adapter():               # reference = base
                logp_ref = token_logprobs(model, ids, device)

        n_tok = int(m.sum().item())
        # policy-gradient surrogate + per-token KL to reference, on assistant
        # tokens only. Backward PER EPISODE so only one graph is live at a time
        # (accumulating over the group would hold k graphs and OOM). Scale by
        # 1/(k*n_tok) here so the accumulated gradient matches a group mean.
        pg = -(a.to(device) * logp * m).sum()
        kl = ((logp - logp_ref) * m).sum()
        loss_e = (pg + kl_coef * kl) / (len(episodes) * max(n_tok, 1))
        loss_e.backward()

        total_loss += loss_e.item()
        total_kl += kl.item()
        total_tokens += n_tok

    if total_tokens == 0:
        return None
    return {"loss": total_loss, "kl": total_kl / max(total_tokens, 1),
            "reward_mean": rewards.mean().item(), "reward_std": rewards.std().item(),
            "solve_rate": sum(e.solved for e in episodes) / len(episodes)}


# --- main loop --------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("instances")
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="qwen-railplan-sft")
    ap.add_argument("--out", default="qwen-railplan-grpo")
    ap.add_argument("--n", type=int, default=64, help="instances to cycle over")
    ap.add_argument("--k", type=int, default=6, help="rollouts per instance")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--kl-coef", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=1.1)
    ap.add_argument("--max-calls", type=int, default=MAX_CALLS)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # inference/generation uses the BASE template (no {% generation %} markers)
    tok.chat_template = AutoTokenizer.from_pretrained(args.base).chat_template

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa",
    )
    # load the SFT adapter as the trainable policy; reference = adapter disabled
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
    model.print_trainable_parameters()
    # trade compute for memory: recompute activations in backward instead of
    # storing them. Essential to fit a long-episode backward pass on 24GB.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    snap = Path(args.snapshot)
    env = Env(World(snap), snap)
    instances = [json.loads(l) for l in Path(args.instances).open()][: args.n]

    log = []
    for step in range(1, args.steps + 1):
        inst = random.choice(instances)

        model.eval()
        episodes = [
            rollout(model, tok, env, inst, args.max_calls,
                    args.temperature, args.max_new, device)
            for _ in range(args.k)
        ]

        model.train()
        opt.zero_grad()
        stats = grpo_step(model, tok, episodes, args.kl_coef, device)
        if stats is None:
            print(f"step {step:>4}  (group flat, skipped)  "
                  f"inst={inst['id']} rewards="
                  f"{[round(e.reward,2) for e in episodes]}")
            continue
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

        log.append(stats)
        print(f"step {step:>4}  loss {stats['loss']:+.3f}  kl {stats['kl']:+.4f}  "
              f"R {stats['reward_mean']:+.2f}±{stats['reward_std']:.2f}  "
              f"solve {stats['solve_rate']:.2f}  calls "
              f"{[e.n_calls for e in episodes]}")

        if step % args.save_every == 0:
            model.save_pretrained(f"{args.out}/step-{step}")
            print(f"  saved {args.out}/step-{step}")

    model.save_pretrained(args.out)
    print(f"\nsaved final adapter to {args.out}")
    # rolling solve-rate so you can see if it moved
    if log:
        first = sum(s["solve_rate"] for s in log[:20]) / min(20, len(log))
        last = sum(s["solve_rate"] for s in log[-20:]) / min(20, len(log))
        print(f"solve rate: first20 {first:.2f} -> last20 {last:.2f}")


if __name__ == "__main__":
    main()