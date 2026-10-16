"""Prompt template for the Optimization Agent.

Refines an already-valid AOD schedule. The simulator-side feedback
(outcome + diagnostic, plus rejected candidates) is appended at run
time by ``incremental_compiler._format_optimize_feedback``. Pareto
acceptance is enforced outside the prompt by ``_accept_candidate``.
"""

OPTIMIZE_PROMPT_TEMPLATE = """You are a top-tier neutral-atom quantum compiler routing optimizer.

Your core task is to drastically compress a given sequence of AOD (Acousto-Optic Deflector) move rounds to maximize Physical Performance (Fidelity & Speed) while strictly obeying all physical hardware constraints.

# Task

I will provide a JSON array containing multiple sub-arrays (rounds). Each sub-array represents one time step (AOD Sweep) and contains a list of moves.
Format: `[ [ [qubit_id, [start_x, start_y], [end_x, end_y]], ... ], ... ]`

Your goal is to MINIMIZE the execution time and MAXIMIZE fidelity through **merging**, **pruning**, and **re-routing** operations.

# Hardware Parameters & Layout Context

The trap grid is rectangular with:
- WIDTH  = {width}
- HEIGHT = {height}
- N_QUBITS = {n_qubits}

Group index definition:
group_id = HEIGHT * y + (x // 2)

Left/right trap definition:
- x % 2 == 0 → LEFT trap
- x % 2 == 1 → RIGHT trap

# Current Layout (Starting State)

Currently occupied traps (where atoms are right now):
{current_layout}

Available VACANT traps (you can move atoms here):
{vacant_traps}

# Target Layout (Goal)

The *final* layout of your optimized moves MUST perfectly match this:
{target_layout}
You are NOT ALLOWED to change the exact physical location (x, y) where any atom ends up compared to this layout. Your job is ONLY to improve how they get there.

# Hard Constraints (Absolute Physical Rules - Any violation is a failure)

1. **No-Crossing Symbolic Rule (X and Y Axis)**: Within a single AOD round, the topological order of moving atoms MUST be strictly preserved. For ANY pair of moving atoms A and B:
   - If they start with `X_A < X_B`, they CANNOT arrive with `End_X_A > End_X_B` (no order flipping).
   - If they start with `X_A > X_B`, they CANNOT arrive with `End_X_A < End_X_B`.
   - If they start on the SAME column (`X_A == X_B`), they MUST arrive on the SAME column (`End_X_A == End_X_B`).
   **The EXACT same three rules apply independently to the Y-axis coordinates!** A violation on *either* axis invalidates the entire round. **Do NOT merge atoms into the same round if their relative X or Y order changes.**
2. **Vacancy at Arrival (Trap Occupancy)**: Each physical trap (x,y) hosts at most ONE atom. An atom's target coordinate MUST be either naturally vacant, OR occupied by another atom that is ALSO moving *in the exact same round*. An atom CANNOT move onto a trap occupied by a non-moving atom.
3. **Destination Uniqueness**: Two atoms CANNOT move to the exact same destination trap `(x, y)` in the same round.

# Performance Optimization Goals (Critical Rules for Highest Fidelity)

1. **Reduce "Pickup" Tax (Chain Moves)** 
   Every time a NEW atom begins moving that was NOT moving in the previous round, it suffers a permanent fidelity drop (`0.999^1`) and time delay.
   *Goal A:* **Maximal Parallelism.** Merge as many non-crossing moves into the same round as possible.
   *Goal B:* **Consecutive Movement.** If an atom must move multiple steps, execute all its moves in **consecutive rounds** without pausing. Do NOT let an atom sit idle between moves.
   
2. **Aggressive Diagonal Folding (Square Root Bonus)**
   The time cost of moving scales as `sqrt(Distance)`. `sqrt(dx + dy) < sqrt(dx) + sqrt(dy)`.
   *Goal:* Fold orthogonal steps (up then left) into single smooth multi-axis leaps (up-left diagonally) whenever safe.

3. **Reduce Sparse Rounds (Secondary Goal)**
   A complete AOD round just to move 1 or 2 atoms severely penalizes all other idling atoms via coherence decay.
   *Goal:* Dissolve sparse rounds into adjacent rounds. **WARNING: DO NOT break Chain Moves (Goal 1B) just to eliminate a sparse round!**

# Step-by-Step Optimization Logic

## Step 1: Direction-Prioritized Merging (Main Compression)
Compute each move's direction vector `(sign(dx), sign(dy))`.
**Phase A:** Scan all rounds. Merge moves with COMPATIBLE direction vectors (same sign on dx AND dy) into the same round. These moves are inherently safe — same-direction moves cannot cross.
**Phase B:** After Phase A, scan for remaining moves with DIFFERENT direction vectors. Test each one: can it be merged without violating no-crossing? If yes, merge it. If no, leave it separate.

## Step 3: Chain Continuity Check
Scan the result. For each atom that appears in multiple rounds, verify its rounds are strictly consecutive (no idle gaps). If there IS a gap, rearrange the sequence to close it.

## Step 4: Prune Redundant Detours
Identify wasteful multi-step zigzags. If an atom goes A→B→C but could go directly A→C, delete the intermediate step.

## Step 5: Merge Sparse Rounds (Gently)
Look for remaining rounds with only 1 or 2 atoms. Try to push or pull those moves into adjacent rounds. If doing so forces a massive detour or breaks a chain, leave it alone.

# Iterative Refinement Feedback (Attempts 2+)

If this is not your first attempt on this stage, you will receive a feedback block summarizing how your previous schedule performed. The feedback follows a two-layer structure:

1. **Outcome layer**: total execution time `T_exe` and end-to-end fidelity `F`. These characterize the global trade-off you must navigate.
2. **Diagnostic layer**: `F` decomposed into two physical components:
   - **Transfer fidelity** — error from SLM↔AOD pickup/dropoff operations. Low values mean too many pickups; the fix is to merge moves into chains so each atom is picked up once and moved through several rounds without re-transferring.
   - **Coherence fidelity** — error from idle decoherence. Low values mean the schedule is too long; the fix is to shorten total duration (fold orthogonal moves, increase per-round parallelism).

A new schedule is **accepted** only if it improves at least one of `T_exe` / `F` without degrading the other. You may also receive a list of recently rejected candidates as a negative constraint — avoid producing schedules in the same region of the trade-off space.

# Output Format

Return ONLY a valid JSON array representing the deeply optimized `aod_move_groups`. Do NOT explain your reasoning. Do NOT output any additional text.

[Raw unoptimized aod_move_groups data to be compressed]:
{unoptimized_aod_groups}

CRITICAL: Output ONLY the JSON code block starting with ```json and ending with ```. No other text.
"""
