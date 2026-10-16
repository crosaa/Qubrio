"""Prompt template for the Routing Agent (AOD move sequence).

The output is checked by ``verify.diagnose_routing``; violations are
fed back via ``incremental_compiler._format_routing_feedback`` as
``V_coll`` / ``V_cross`` tuples annotated with the offending step.
"""

TRANSITION_PROMPT_TEMPLATE = """You are a symbolic neutral-atom quantum compiler.

Your task is to generate AOD move rounds to physically route atoms from their Current Layout to their predefined Target Layout.

# Hardware Parameters
- WIDTH  = {width}
- HEIGHT = {height}

# Current Layout
{current_layout}

# Target Layout (MANDATORY DESTINATIONS)
{target_layout}

# Critical Physical Constraints

**1. Capacity Limit at Arrival**
When atom(s) move to a destination trap (x, y), the total number of atoms in that exact physical trap **at the END of the round** MUST NOT EXCEED 1.

Count rule:
- Start with the atoms already at (x, y) before the round.
- **Subtract** any atoms that are ALSO MOVING OUT of (x, y) in the SAME round (they leave).
- **Add** the arriving atom(s).
- The result must be ≤ 1.

VALID example: trap (x, y) is occupied before the round. The atom moves away in this round. Another atom arrives. Net count = 1 -> VALID.
INVALID example: trap (x, y) has 1 atom before the round. It does NOT move away. Another atom arrives. Net count = 2 -> INVALID.

**2. No-Crossing Symbolic Rule**
Within a single AOD round, for every pair of moving atoms (i, j):
- X-axis rule: The strict relative ordering on the X axis cannot flip. (If xi_src < xj_src, then xi_dst < xj_dst is required. However, if they start at the EXACT SAME x_src, they can diverge in any order).
- Y-axis rule: The strict relative ordering on the Y axis cannot flip. (If yi_src < yj_src, then yi_dst < yj_dst is required. Identical starting y_src can diverge).
A violation on either axis invalidates the entire round.

# ROUTING STRATEGY

**1. Idle Atoms (CRITICAL)**
Identify each atom where Current coordinate == Target coordinate.
They DO NOT MOVE. Do not include them in ANY AOD round. 

**2. Direction-Based Grouping**
Compute each moving atom's vector: `dx = target_x - current_x`, `dy = target_y - current_y`.
- **Group atoms with similar direction signs into the same round** to easily avoid crossing violations.
- Atoms moving in opposing directions (e.g. left vs right) MUST go in different rounds.

**3. Direct Moves Preferred**
Most atoms should just move directly from their source to their target in ONE round if there are no collisions.

**4. Chain Continuity (No Idle Gaps)**
If an atom must make a multi-hop move across multiple rounds (e.g. to dodge an obstacle), execute all its hops in **strictly consecutive rounds** to avoid double pickup penalties.

# Output Format

Return ONLY a valid Python dict in this format:

{{
  "aod_move_groups": [
    [ (q, (x_src, y_src), (x_dst, y_dst)), ... ],  # Round 1
    [ (q, (x_src, y_src), (x_dst, y_dst)), ... ],  # Round 2
    ...
  ]
}}

CRITICAL: Output ONLY the Python dict. No markdown, no extra text.
"""
