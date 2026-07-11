# constrained-railplan

An agentic RL environment for constrained itinerary planning over real GTFS data, with verifiable rewards and Lagrangian constraint enforcement.

A policy receives an underspecified natural-language travel request, searches a timetable through local tool calls, and submits an itinerary. Feasibility and constraint satisfaction are checked exactly by replaying the itinerary against GTFS. Hard constraints are enforced with learned Lagrange multipliers rather than hand-tuned penalty weights.

## Why

Post-training with verifiable rewards (RLVR) needs a domain where checking is exact and cheap but producing is hard. Public transit timetables provide this: a proposed itinerary is trivially validated (do the legs chain? do the trips run today? is the transfer time respected?), while producing one under vague, competing user constraints is not.

The routing itself is *not* the contribution — Dijkstra over the timetable graph solves that exactly, and is used here as the verifier. The contribution is:

1. **The environment.** Only local lookup tools are exposed (`departures`, `leg`, `find_station`). No `plan()` tool. Search, backtracking, and stopping are the policy's job.
2. **Constraints as constraints.** A request like *"arrive before evening, at most two changes, avoid Belgrade"* defines a CMDP, not a scalarized objective. Training maximizes feasibility subject to `E[C_i] ≤ 0` via dual ascent on λ, instead of folding constraints into a weighted penalty.
3. **Shadow prices.** At convergence `λ_i` is the shadow price of constraint `i` (Envelope Theorem: `∂V*/∂θ_i = −λ*_i`). When no itinerary satisfies every constraint, the multipliers rank which constraint to relax first — a principled answer to a problem where a solver just returns `UNSAT`.

## Claim under test

Fixed penalty weights either over-satisfy (the agent avoids itineraries near a constraint boundary) or get hacked (the agent violates a cheap constraint to satisfy an expensive one). Adaptive multipliers track the threshold. Deliverable: λ trajectories plotted against per-constraint satisfaction rate, versus a fixed-weight baseline at several weight settings.

## Design

**Tools**

```
find_station(query)                  -> [{station_id, name, coords}]
departures(station_id, after, date)  -> [{trip_id, headsign, dep, arr}]
leg(from_id, to_id, date, after)     -> [{trip_id, dep, arr}]        # direct legs only
submit(legs: [leg_id])               -> terminates the episode
```

Episodes run ~5–20 tool calls. Tool-call budget is penalized so the policy cannot brute-force the timetable.

**Verifier**

`replay(itinerary, date) -> FeasibilityReport`, returning exact feasibility (leg chaining, min transfer time, service calendar) and per-constraint costs (transfers used, arrival time, forbidden stations visited). This *is* the reward function. It is built and tested before any model is trained.

**Reward**

```
maximize   E[ feasible ] - c · (tool calls)
subject to E[ C_i ] ≤ 0     for each hard constraint i

L(π, λ) = E[ feasible ] - Σ_i λ_i · C_i        # dual ascent on λ
```

**Instance generation (backwards)**

1. Sample `(origin, destination, date, constraint set)` from the feed.
2. Solve with the reference solver. Keep the instance iff it is feasible **and** the unconstrained-optimal itinerary violates at least one constraint. Without this second condition the constraints are decorative and λ has nothing to do.
3. Paraphrase the constraint set into natural language at varying levels of vagueness ("before evening", "a couple of changes at most").

Ground-truth constraints and reference solutions come for free, at unlimited scale.

## Build order

1. **Freeze scope.** One national feed (small, with real branching), one service date, cached to parquet. Deterministic and reproducible.
2. **Verifier.** Tested against known-good itineraries and deliberately broken ones. If it cannot catch a 3-minute impossible transfer, nothing downstream is meaningful.
3. **Instance generator**, with the non-triviality filter above.
4. **Headroom check (go/no-go).** Hand the tools to a strong prompted model on ~50 instances with no training. Measure solve rate.
   - `> 80%` — no headroom, the tasks are too easy. Harden them.
   - `< 10%` — the tool interface or the horizon is broken, not the policy.
   - `20–50%` — post-training has something to learn. Proceed.
5. **SFT warm start.** ~500 successful trajectories, converted from reference solutions into tool-call traces. Cold-start GRPO on tool use rarely gets off the ground.
6. **GRPO + dual ascent on λ.** Qwen2.5-3B-Instruct, LoRA, single GPU.

## Contamination and reward hacking

Held out by *constraint combination*, not by city pair. A policy that memorizes the solver's default output for a route would otherwise score well on unseen constraints for a seen route.

Known hacks to watch for and report:

- `submit` on a trivially short feasible itinerary. Mitigated by rejecting origin/destination pairs where the trivial solution is admissible.
- Satisfying constraints by ignoring the request and emitting the unconstrained optimum. Caught by the non-triviality filter at generation time.

## Non-goals

This is not an itinerary product. A prompted model emitting a constraint JSON, plus the existing solver, would serve users better with no training loop at all. The trained policy is an object of study, not a deployment artifact.

## Status

Scoping.
