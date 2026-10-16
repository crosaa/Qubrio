"""Prompt template for the initial qubit-to-trap placement (stage 1)."""

INITIAL_LAYOUT_PROMPT_TEMPLATE = """You are a top-tier neutral-atom quantum compiler.

Your task is to generate a physically valid INITIAL LAYOUT for a neutral-atom quantum computer that MAXIMIZES performance for future routing.

# Hardware Parameters

The trap grid is rectangular with:
- WIDTH  = {width}
- HEIGHT = {height}
- N_QUBITS = {n_qubits}

Trap coordinates are integer lattice points:
- x ∈ {{0, 1, ..., WIDTH-1}}
- y ∈ {{0, 1, ..., HEIGHT-1}}

# Trap Capacity and CZ Execution

1. Each trap coordinate (x, y) can physically hold a MAXIMUM of 1 atom.
2. Left and Right Traps: x % 2 == 0 is a LEFT trap, x % 2 == 1 is a RIGHT trap.
3. Group Definition: `group_id = HEIGHT * y + (x // 2)`. A group contains exactly one left trap and one right trap.
4. Two atoms occupying the SAME group (one in left trap, one in right trap) execute a CZ gate under global illumination.

# First CZ Stage

The circuit will execute the following CZ gates as Stage 1:

{cz_stage1}

# Initial Layout Constraints

Your initial layout must satisfy:

1. **Complete coverage:** Each qubit 0..(N_QUBITS-1) must be assigned to exactly one trap coordinate.
2. **Capacity Limit:** Each physical trap (x, y) can hold at most 1 atom.
3. **Left/Right Validity:** Atoms can be placed in either left (even x) or right (odd x) traps.
4. **No Pre-Pairing Required:** You do NOT need to pair atoms for CZ Stage 1 in the initial layout. Generating a simple, ordered grid layout (e.g., filling row by row) is acceptable and often preferred. The first AOD stage will handle the routing for CZ Stage 1.
5. **Bounds:** All coordinates must satisfy 0 <= x < WIDTH and 0 <= y < HEIGHT.

# PERFORMANCE OPTIMIZATION STRATEGY (CRITICAL)

The time it takes to execute the entire quantum circuit depends heavily on this initial layout! Every time atoms move apart or together, the penalty scales with the Manhattan Distance grid distance.

**Optimization Goal: DENSE CLUSTERING**
- Do NOT spread the atoms out randomly across the entire WIDTH x HEIGHT grid.
- **Pack all atoms as tightly together as possible** into a solid rectangular block or dense square near the center or a corner of the grid (e.g. x=0..4, y=0..3).
- **Proximity:** If Qubit A and Qubit B will need to interact soon, try to place their initial groups physically adjacent to each other.
- **Minimize Empty Space:** Leave as few empty traps *between* the atoms as possible. The more compact the initial layout, the shorter the distance atoms have to travel in later AOD rounds.

# Output Format

Return ONLY a valid Python dict in this exact format (no markdown, no extra text):

{{
  "initial_layout": {{
    0: (x, y),
    1: (x, y),
    2: (x, y),
    ...
    {n_qubits_minus_1}: (x, y)
  }}
}}

Each key is a qubit index (int), each value is a coordinate tuple (x, y).

CRITICAL: Output ONLY the Python dict. No explanations, no markdown code blocks, no extra text.
"""

