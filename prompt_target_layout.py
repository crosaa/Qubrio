"""Prompt template for the Placement Agent (per-stage target layout).

The output is checked by ``verify.diagnose_layout``; violations are
fed back via ``incremental_compiler._format_placement_feedback``.
"""

TARGET_LAYOUT_PROMPT_TEMPLATE = """You are a top-tier neutral-atom quantum compiler.

Your task is to generate the optimal TARGET LAYOUT for atoms transitioning from CZ Stage {stage_from} to CZ Stage {stage_to}.

# Hardware Parameters
- WIDTH  = {width}
- HEIGHT = {height}
- N_QUBITS = {n_qubits}

# Current Layout
{current_layout}

# Target CZ Stage {stage_to} Pairing
The next CZ stage requires the following pairs of atoms to share a group:
{cz_stage_next}
Each pair MUST occupy the same group_id. `group_id = HEIGHT * y + (x // 2)`. One atom in the left trap (even x), one in the right trap (odd x).

# Future CZ Stages (God's Eye View / Lookahead)
To help you make smarter decisions, here are the pairings for the subsequent stages:
{future_stages}
Pre-fetching (Optional but encouraged): If you notice that two atoms will be paired in a future stage, you may optionally move them closer together *now* (in this stage), provided it does not compromise the immediate {stage_to} pairings.

# Constraints
1. Complete assignment: Every qubit 0 to {n_qubits_minus_1} must have exactly one (x, y) coordinate.
2. Capacity Limit: Each trap coordinate can hold a MAXIMUM of 1 atom at all times.
3. CZ Pairs MUST share the exact same group_id (same x//2 and y).
4. UNINVOLVED ATOMS (SOFTENED LAZINESS): Atoms NOT listed in the CZ Stage {stage_to} pairs do not strictly need to move, BUT you are ENCOURAGED to move them if they are blocking the path of other atoms, or if you want to pre-fetch them closer to their future pairing partners for upcoming stages.

# OPTIMIZATION: Global Shortest Path & Min-Max Distance
Assign pairs to vacant coordinates to achieve a GLOBAL optimum, not a greedy local one. 
1. Primary Goal (Min-Max): Minimize the MAXIMUM distance any individual atom has to travel. Avoid extreme outliers where one atom travels 6+ units while others travel 1 unit.
2. Secondary Goal: Minimize the TOTAL sum of Manhattan distances across all moving atoms. 

# Output Format
Return ONLY a valid Python dict mapping qubit IDs to (x, y) coordinates:
{{
  "target_layout": {{
    0: (x, y),
    ...
    19: (x, y)
  }}
}}

CRITICAL: Output ONLY the Python dict. No explanations, no markdown blocks, no extra text.
"""
