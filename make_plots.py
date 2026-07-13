"""
make_plots.py -- generate report figures from the experiment's probe/log files.

    python make_plots.py           # writes figures/*.png

Reads only files produced during the runs; each plot degrades gracefully if its
inputs are missing. matplotlib only, no seaborn.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path("figures")
FIG.mkdir(exist_ok=True)

# muted, report-friendly palette
C0, C1, C2 = "#3b6ea5", "#c1666b", "#6a994e"
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                     "axes.axisbelow": True, "grid.alpha": 0.3})


def load(path):
    p = Path(path)
    if not p.exists():
        return None
    return [json.loads(l) for l in p.open() if l.strip()]


def group_by_instance(rows):
    by = defaultdict(list)
    for r in rows:
        by[r.get("instance_id", r.get("id"))].append(r)
    return by


# --- Plot 1: reward spread, without vs with potential-based shaping ----------

def plot_shaping():
    no = load("qwen_base.jsonl")             # base, no shaping
    yes = load("qwen_base_coords.jsonl")     # base + distance shaping
    if not no or not yes:
        print("skip plot 1: missing qwen_base / qwen_base_coords")
        return

    def spreads(rows):
        out = []
        for _, g in group_by_instance(rows).items():
            rw = [x["reward"] for x in g if "reward" in x]
            if len(rw) >= 2:
                out.append(max(rw) - min(rw))
        return out

    a, b = spreads(no), spreads(yes)
    if not a or not b:
        print("skip plot 1: not enough multi-rollout instances")
        return

    fig, ax = plt.subplots(figsize=(5.2, 4))
    bp = ax.boxplot([a, b], labels=["no shaping", "distance\nshaping"],
                    widths=0.5, patch_artist=True, showmeans=True)
    for patch, c in zip(bp["boxes"], [C1, C2]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_ylabel("per-instance reward spread\n(max - min across rollouts)")
    ax.set_title("Potential-based shaping restores GRPO's gradient")
    import statistics as st
    ax.text(1, max(a) if a else 0, f" median {st.median(a):.3f}", va="bottom", fontsize=9)
    ax.text(2, max(b) if b else 0, f" median {st.median(b):.3f}", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "1_shaping_reward_spread.png"); plt.close(fig)
    print("wrote figures/1_shaping_reward_spread.png")


# --- Plot 2: feasibility / solve rate by transfer depth ----------------------

def plot_depth():
    inst = load("instances.jsonl")
    probe = load("probe_sft3.jsonl") or load("probe_shallow.jsonl")
    if not inst or not probe:
        print("skip plot 2: missing instances / probe")
        return
    # join by position (probe iterates the instance file in order); fall back to id
    depth_of = {i.get("id", n): i["reference"]["transfers"] for n, i in enumerate(inst)}
    by_depth = defaultdict(lambda: {"n": 0, "feas": 0, "solv": 0})
    for n, p in enumerate(probe):
        d = depth_of.get(p.get("instance_id", n))
        if d is None:
            continue
        s = by_depth[d]
        s["n"] += 1
        s["feas"] += bool(p.get("feasible"))
        s["solv"] += bool(p.get("solved"))
    depths = sorted(by_depth)
    if not depths:
        print("skip plot 2: no depth join")
        return
    feas = [by_depth[d]["feas"] / by_depth[d]["n"] for d in depths]
    solv = [by_depth[d]["solv"] / by_depth[d]["n"] for d in depths]
    ns = [by_depth[d]["n"] for d in depths]

    fig, ax = plt.subplots(figsize=(5.6, 4))
    x = range(len(depths))
    w = 0.38
    ax.bar([i - w/2 for i in x], feas, w, label="feasible", color=C0, alpha=0.8)
    ax.bar([i + w/2 for i in x], solv, w, label="solved", color=C2, alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{d} transf.\n(n={n})" for d, n in zip(depths, ns)])
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1)
    ax.set_title("Feasibility collapses with transfer depth")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "2_depth_breakdown.png"); plt.close(fig)
    print("wrote figures/2_depth_breakdown.png")


# --- Plot 3: GRPO reward trajectory vs solve rate ----------------------------

def plot_grpo():
    p = Path("grpo.log")
    if not p.exists():
        print("skip plot 3: missing grpo.log")
        return
    steps, R, sd, solve = [], [], [], []
    pat = re.compile(r"step\s+(\d+)\s+loss\s+[-+][\d.]+\s+kl\s+[-+][\d.]+\s+"
                     r"R\s+([-+][\d.]+)±([\d.]+)\s+solve\s+([\d.]+)")
    for line in p.open():
        m = pat.search(line)
        if m:
            steps.append(int(m.group(1)))
            R.append(float(m.group(2)))
            sd.append(float(m.group(3)))
            solve.append(float(m.group(4)))
    if not steps:
        print("skip plot 3: no step lines parsed")
        return
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(steps, R, color=C0, label="mean reward")
    ax1.fill_between(steps, [r - s for r, s in zip(R, sd)],
                     [r + s for r, s in zip(R, sd)], color=C0, alpha=0.15)
    ax1.set_xlabel("GRPO step"); ax1.set_ylabel("mean reward", color=C0)
    ax1.tick_params(axis="y", labelcolor=C0)
    ax2 = ax1.twinx()
    ax2.plot(steps, solve, color=C1, label="solve rate")
    ax2.set_ylabel("solve rate", color=C1); ax2.set_ylim(-0.02, 1)
    ax2.tick_params(axis="y", labelcolor=C1)
    ax1.set_title("GRPO optimizes the shaped reward, but solves stay at zero")
    fig.tight_layout(); fig.savefig(FIG / "3_grpo_trajectory.png"); plt.close(fig)
    print("wrote figures/3_grpo_trajectory.png")


# --- Plot 4: verify-before-submit rate across iterations ---------------------

def verified_rate(rows):
    if not rows:
        return None
    vs = tot = 0
    for x in rows:
        if not x.get("submitted"):
            continue
        tot += 1
        conf = set()
        for s in x.get("trace", []):
            if s.get("tool") == "leg" and "error" not in (s.get("result") or {}):
                a = s.get("args", {})
                conf.add((a.get("trip_id"), a.get("board"), a.get("alight")))
        for s in x.get("trace", []):
            if s.get("tool") == "submit":
                legs = [(l.get("trip_id"), l.get("board"), l.get("alight"))
                        for l in s.get("args", {}).get("legs", [])]
                if legs and all(l in conf for l in legs):
                    vs += 1
                break
    return vs / tot if tot else None


def plot_verification():
    stages = [("base", "qwen_base_coords.jsonl"),
              ("SFT v1", "probe_sft2.jsonl"),
              ("SFT v2\n(A+B)", "probe_sft3.jsonl")]
    labels, rates = [], []
    for name, f in stages:
        r = verified_rate(load(f))
        if r is not None:
            labels.append(name); rates.append(r)
    if len(rates) < 2:
        print("skip plot 4: not enough stages with trace data")
        return
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.bar(labels, rates, color=[C1, C0, C2][:len(rates)], alpha=0.8)
    ax.set_ylabel("fraction of submissions\nwith all legs leg()-verified")
    ax.set_ylim(0, 1)
    ax.set_title("Verification behavior across training iterations")
    for i, v in enumerate(rates):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "4_verification_rate.png"); plt.close(fig)
    print("wrote figures/4_verification_rate.png")


if __name__ == "__main__":
    plot_shaping()
    plot_depth()
    plot_grpo()
    plot_verification()
    print("done. figures in", FIG.resolve())
