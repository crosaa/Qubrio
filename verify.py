"""Deterministic verifiers for placement and routing.

Public surface:
    diagnose_layout   — returns {V_coll, V_adj, V_xtalk, fatal}
    diagnose_routing  — returns {V_coll, V_cross, fatal}
    check_aod_round   — raises on the first violation in a single round
    verify_output     — full-schedule sanity check

Hardware model: split-trap grid with WIDTH = 2·⌈√n⌉, HEIGHT = ⌈√n⌉.
A "group" is a (left, right) trap pair sharing the same (y, x//2);
the two atoms in a group can execute a CZ gate.
"""

from __future__ import annotations
from dataclasses import dataclass
from math import ceil, sqrt
from typing import Dict, Tuple, List, Any, Set

Coord = Tuple[int, int]
Qubit = int
Pair = Tuple[int, int]
Move = Tuple[int, Coord, Coord]      # (q, src, dst)
AODRound = List[Move]                # list of moves
AODMoveGroups = List[AODRound]       # list of rounds

# ----------------------------
# Helpers: circuit + hardware
# ----------------------------

def normalize_pair(a: int, b: int) -> Pair:
    return (a, b) if a < b else (b, a)

def stage_pair_set(stage: List[Pair]) -> Set[Pair]:
    return {normalize_pair(a, b) for a, b in stage}

def infer_n_qubits(cz_stages: List[List[Pair]]) -> int:
    mx = -1
    for stage in cz_stages:
        for a, b in stage:
            mx = max(mx, a, b)
    if mx < 0:
        raise ValueError("cz_stages is empty; cannot infer number of qubits.")
    return mx + 1

@dataclass(frozen=True)
class Hardware:
    n_qubits: int
    width: int
    height: int

    @staticmethod
    def from_cz_stages(cz_stages: List[List[Pair]]) -> "Hardware":
        n = infer_n_qubits(cz_stages)
        side = ceil(sqrt(n))
        return Hardware(n_qubits=n, width=2 * side, height=side)

    def in_bounds(self, c: Coord) -> bool:
        x, y = c
        return 0 <= x < self.width and 0 <= y < self.height

    def group_id(self, c: Coord) -> int:
        """Left trap (even x) and right trap (odd x) share the same group."""
        x, y = c
        return self.height * y + (x // 2)

# ----------------------------
# Helpers: layout / occupancy
# ----------------------------

def invert_layout(layout: Dict[int, Coord]) -> Dict[Coord, List[int]]:
    inv: Dict[Coord, List[int]] = {}
    for q, c in layout.items():
        if c not in inv:
            inv[c] = []
        inv[c].append(q)
        if len(inv[c]) > 1:
            raise ValueError(f"Trap collision: {c} has more than 1 atom: {inv[c]}. "
                             f"Each physical trap (left or right) holds exactly 1 atom.")
    return inv

def parse_layout(initial_layout: Any) -> Dict[int, Coord]:
    if not isinstance(initial_layout, dict):
        raise ValueError("'initial_layout' must be a dict {q: (x,y)}.")
    parsed: Dict[int, Coord] = {}
    for k, v in initial_layout.items():
        q = int(k)  # tolerate JSON string keys
        if not (isinstance(v, (tuple, list)) and len(v) == 2):
            raise ValueError(f"Layout coord must be (x,y) for q{q}, got {v}")
        parsed[q] = (int(v[0]), int(v[1]))
    return parsed

def parse_move_groups(item: Any) -> AODMoveGroups:
    if not (isinstance(item, dict) and "aod_move_groups" in item):
        raise ValueError("Expected {'aod_move_groups': [...]} between CZ stages.")
    groups = item["aod_move_groups"]
    if not isinstance(groups, list):
        raise ValueError("'aod_move_groups' must be a list of AOD rounds.")
    parsed: AODMoveGroups = []
    for rnd in groups:
        if not isinstance(rnd, list):
            raise ValueError("Each AOD round must be a list of moves.")
        moves: AODRound = []
        for m in rnd:
            if not (isinstance(m, (tuple, list)) and len(m) == 3):
                raise ValueError(f"Move must be (q, src, dst), got {m}")
            q, src, dst = m
            if not (isinstance(src, (tuple, list)) and len(src) == 2):
                raise ValueError(f"Move src must be (x,y), got {src}")
            if not (isinstance(dst, (tuple, list)) and len(dst) == 2):
                raise ValueError(f"Move dst must be (x,y), got {dst}")
            moves.append((int(q), (int(src[0]), int(src[1])), (int(dst[0]), int(dst[1]))))
        parsed.append(moves)
    return parsed

def parse_cz_stage(item: Any) -> List[Pair]:
    if not (isinstance(item, dict) and "cz_stage" in item):
        raise ValueError("Expected {'cz_stage': [...]} item.")
    cz = item["cz_stage"]
    if not isinstance(cz, list):
        raise ValueError("'cz_stage' must be a list of pairs.")
    out: List[Pair] = []
    for p in cz:
        if not (isinstance(p, (tuple, list)) and len(p) == 2):
            raise ValueError(f"CZ pair must be (q0,q1), got {p}")
        out.append((int(p[0]), int(p[1])))
    return out

# ----------------------------
# CZ group audit
# ----------------------------

def audit_groups_pre_cz(hw: Hardware, layout: Dict[int, Coord], cz_stage: List[Pair], stage_idx: int) -> None:
    allowed = stage_pair_set(cz_stage)
    groups: Dict[int, List[Tuple[int, Coord]]] = {}
    for q, c in layout.items():
        if not hw.in_bounds(c):
            raise ValueError(f"[CZ{stage_idx}] q{q} out of bounds at {c}")
        gid = hw.group_id(c)
        groups.setdefault(gid, []).append((q, c))

    for gid, members in groups.items():
        if len(members) <= 1:
            continue
        if len(members) > 2:
            raise ValueError(f"[CZ{stage_idx}] group {gid} has >2 atoms: {members}")

        (q1, c1), (q2, c2) = members

        pair = normalize_pair(q1, q2)
        if pair not in allowed:
            raise ValueError(f"[CZ{stage_idx}] unintended 2-atom group {gid}: pair {pair} not in stage.")


def diagnose_layout(
    hw: Hardware,
    layout: Dict[int, Coord],
    cz_stage: List[Pair],
) -> Dict[str, Any]:
    """Diagnose a candidate layout against ``cz_stage``.

    Returns ``{V_coll, V_adj, V_xtalk, fatal}``. Each categorical set is
    a list of tuples ``(q_i, q_j, P(q_i), P(q_j))``:
      V_coll   same trap, or >2 atoms in one group
      V_adj    CZ pairs not co-located
      V_xtalk  non-CZ pairs sharing a group
    ``fatal`` covers structural errors (out-of-bounds, missing qubits)
    that must be fixed before categorical feedback is meaningful.
    All four are empty for a valid layout.
    """
    V_coll: List[Tuple[int, int, Coord, Coord]] = []
    V_adj: List[Tuple[int, int, Coord, Coord]] = []
    V_xtalk: List[Tuple[int, int, Coord, Coord]] = []
    fatal: List[str] = []

    # 1. Completeness + bounds (fatal-class).
    expected = set(range(hw.n_qubits))
    got = set(layout.keys())
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    if missing:
        fatal.append(f"Missing qubits in layout: {missing}")
    if extra:
        fatal.append(f"Unknown qubits in layout: {extra}")

    for q, c in layout.items():
        if not hw.in_bounds(c):
            fatal.append(f"q{q} out of bounds at {c}")

    if fatal:
        # Geometry checks below assume well-formed inputs.
        return {"V_coll": V_coll, "V_adj": V_adj, "V_xtalk": V_xtalk, "fatal": fatal}

    # 2. V_coll: identical (x, y) trap collisions.
    coord_to_qs: Dict[Coord, List[int]] = {}
    for q, c in layout.items():
        coord_to_qs.setdefault(c, []).append(q)
    for c, qs in coord_to_qs.items():
        if len(qs) > 1:
            qs_sorted = sorted(qs)
            for i in range(len(qs_sorted)):
                for j in range(i + 1, len(qs_sorted)):
                    V_coll.append((qs_sorted[i], qs_sorted[j], c, c))

    # 3. Group analysis: V_coll for overcrowded groups, V_xtalk for unintended pairs.
    groups: Dict[int, List[Tuple[int, Coord]]] = {}
    for q, c in layout.items():
        gid = hw.group_id(c)
        groups.setdefault(gid, []).append((q, c))

    allowed = stage_pair_set(cz_stage)

    for gid, members in groups.items():
        if len(members) <= 1:
            continue
        if len(members) > 2:
            # Overcrowded group: every pair is a collision-class violation.
            members_sorted = sorted(members, key=lambda m: m[0])
            for i in range(len(members_sorted)):
                for j in range(i + 1, len(members_sorted)):
                    q1, c1 = members_sorted[i]
                    q2, c2 = members_sorted[j]
                    # Skip duplicates of exact-trap collisions already recorded.
                    if c1 != c2:
                        V_coll.append((q1, q2, c1, c2))
        else:
            (q1, c1), (q2, c2) = members
            pair = normalize_pair(q1, q2)
            if pair not in allowed:
                a, b = pair
                ca, cb = (c1, c2) if a == q1 else (c2, c1)
                V_xtalk.append((a, b, ca, cb))

    # --- 4. V_adj: CZ pair not co-located in the same group ---
    qubit_to_group = {q: hw.group_id(c) for q, c in layout.items()}
    for q1, q2 in cz_stage:
        a, b = normalize_pair(q1, q2)
        ga = qubit_to_group.get(a)
        gb = qubit_to_group.get(b)
        if ga is None or gb is None:
            continue  # missing qubit already in fatal
        if ga != gb:
            V_adj.append((a, b, layout[a], layout[b]))

    return {"V_coll": V_coll, "V_adj": V_adj, "V_xtalk": V_xtalk, "fatal": fatal}

def check_initial_layout(hw: Hardware, layout: Dict[int, Coord]) -> None:
    """Check qubit completeness, bounds, and trap capacity (≤1 per trap)."""
    expected = set(range(hw.n_qubits))
    got = set(layout.keys())
    if got != expected:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        raise ValueError(f"Initial layout qubit set mismatch. Missing={missing}, Extra={extra}")

    for q, c in layout.items():
        if not hw.in_bounds(c):
            raise ValueError(f"Initial layout out of bounds: q{q}@{c}")
    invert_layout(layout)  # raises if any trap has >1 atom

# ----------------------------
# AOD symbolic checks
# ----------------------------

def sign_order(a: int, b: int) -> str:
    if a < b: return "<"
    if a > b: return ">"
    return "="

def violates_no_cross(src_ord: str, dst_ord: str) -> bool:
    # strict order cannot flip
    if src_ord == "<" and dst_ord == ">":
        return True
    if src_ord == ">" and dst_ord == "<":
        return True
    return False

def check_aod_round(hw: Hardware, layout_before: Dict[int, Coord], rnd: AODRound, link_name: str) -> None:
    if not rnd:
        raise ValueError(f"{link_name}: empty AOD round is not allowed.")

    occupied = invert_layout(layout_before)  # coord -> [q]

    # 1. Movers must be unique within a round.
    moving: Set[int] = set()
    for q, src, dst in rnd:
        if q in moving:
            raise ValueError(f"{link_name}: q{q} moved twice in same round.")
        moving.add(q)

    # 2. Per-move sanity + post-round occupancy check.
    expected_occupancy: Dict[Coord, List[int]] = {}
    for c, qs in occupied.items():
        expected_occupancy[c] = [q_ for q_ in qs if q_ not in moving]

    move_map: Dict[int, Tuple[Coord, Coord]] = {}

    for q, src, dst in rnd:
        if q not in layout_before:
            raise ValueError(f"{link_name}: move references unknown q{q}.")

        if layout_before[q] != src:
            raise ValueError(f"{link_name}: src mismatch for q{q}. layout={layout_before[q]}, move={src}")

        if not hw.in_bounds(dst):
            raise ValueError(f"{link_name}: dst out of bounds for q{q}: {dst}")

        if dst not in expected_occupancy:
            expected_occupancy[dst] = []
        expected_occupancy[dst].append(q)

        if len(expected_occupancy[dst]) > 1:
            raise ValueError(
                f"{link_name}: vacancy violated. q{q} dst {dst} will have > 1 atom: {expected_occupancy[dst]}\n"
                f"  REQUIRED FIX: Each physical trap holds at most 1 atom. "
                f"Use the other trap in the same group (left↔right) or choose a different destination."
            )

        move_map[q] = (src, dst)

    # 3. Pairwise symbolic no-crossing among movers.
    M = sorted(moving)
    for i in range(len(M)):
        for j in range(i + 1, len(M)):
            qi, qj = M[i], M[j]
            si, di = move_map[qi]
            sj, dj = move_map[qj]

            x_src = sign_order(si[0], sj[0])
            y_src = sign_order(si[1], sj[1])
            x_dst = sign_order(di[0], dj[0])
            y_dst = sign_order(di[1], dj[1])

            if violates_no_cross(x_src, x_dst) or violates_no_cross(y_src, y_dst):
                raise ValueError(
                    f"{link_name}: symbolic no-crossing violated for moving pair (q{qi},q{qj}) "
                    f"x {x_src}->{x_dst}, y {y_src}->{y_dst}\n"
                    f"  Conflict: q{qi}: {si}→{di}, q{qj}: {sj}→{dj}\n"
                    f"  REQUIRED FIX: Move q{qi} and q{qj} in DIFFERENT rounds, OR use intermediate positions to avoid crossing.\n"
                    f"  Explanation: Paths cross when atoms swap their relative order on x or y axis."
                )

def apply_aod_round(layout: Dict[int, Coord], rnd: AODRound) -> Dict[int, Coord]:
    new_layout = dict(layout)
    for q, src, dst in rnd:
        new_layout[q] = dst
    invert_layout(new_layout)  # raises if a trap ends up with >1 atom
    return new_layout


def diagnose_routing(
    hw: Hardware,
    initial_layout: Dict[int, Coord],
    aod_groups: List[AODRound],
) -> Dict[str, Any]:
    """Walk the AOD sequence and report structured violations.

    Returns ``{V_coll, V_cross, fatal}``. Each categorical entry is a
    tuple ``(q_i, q_j, P^t, P^{t+1}, t)``:
      V_coll   atoms collide after action a_t
      V_cross  a_t flips the x/y ordering of two movers
    The walk stops at the first round with any V_coll, V_cross, or
    fatal error since downstream P^t become undefined. Same constraints
    as ``check_aod_round`` but returns sets instead of raising.
    """
    V_coll: List[Tuple[int, int, Dict[int, Coord], Dict[int, Coord], int]] = []
    V_cross: List[Tuple[int, int, Dict[int, Coord], Dict[int, Coord], int]] = []
    fatal: List[str] = []

    layout = dict(initial_layout)

    for t, rnd in enumerate(aod_groups):
        # Snapshot P^t before applying a_t.
        P_t = dict(layout)

        if not rnd:
            fatal.append(f"step t={t}: empty AOD round")
            return {"V_coll": V_coll, "V_cross": V_cross, "fatal": fatal}

        # 1. Movers must be unique within the round.
        moving: Set[int] = set()
        dup = False
        for q, _src, _dst in rnd:
            if q in moving:
                fatal.append(f"step t={t}: q{q} moved twice in same round")
                dup = True
                break
            moving.add(q)
        if dup:
            return {"V_coll": V_coll, "V_cross": V_cross, "fatal": fatal}

        # 2. Per-move sanity + post-round occupancy.
        expected_occupancy: Dict[Coord, List[int]] = {}
        for q, c in P_t.items():
            if q not in moving:
                expected_occupancy.setdefault(c, []).append(q)

        move_map: Dict[int, Tuple[Coord, Coord]] = {}
        round_fatal = False
        for q, src, dst in rnd:
            if q not in P_t:
                fatal.append(f"step t={t}: move references unknown q{q}")
                round_fatal = True
                continue
            if P_t[q] != src:
                fatal.append(
                    f"step t={t}: src mismatch for q{q} "
                    f"(layout={P_t[q]}, move={src})"
                )
                round_fatal = True
                continue
            if not hw.in_bounds(dst):
                fatal.append(f"step t={t}: dst out of bounds for q{q} at {dst}")
                round_fatal = True
                continue
            expected_occupancy.setdefault(dst, []).append(q)
            move_map[q] = (src, dst)

        if round_fatal:
            return {"V_coll": V_coll, "V_cross": V_cross, "fatal": fatal}

        # Compute P^{t+1} (only valid because all moves passed sanity above).
        P_t1 = dict(P_t)
        for q, _src, dst in rnd:
            P_t1[q] = dst

        # V_coll: any trap holding >1 atom after the move.
        for c, qs in expected_occupancy.items():
            if len(qs) > 1:
                qs_sorted = sorted(qs)
                for i in range(len(qs_sorted)):
                    for j in range(i + 1, len(qs_sorted)):
                        V_coll.append((qs_sorted[i], qs_sorted[j], P_t, P_t1, t))

        # V_cross: pairwise no-crossing among movers.
        M = sorted(moving & set(move_map))
        for i in range(len(M)):
            for j in range(i + 1, len(M)):
                qi, qj = M[i], M[j]
                si, di = move_map[qi]
                sj, dj = move_map[qj]
                if (
                    violates_no_cross(sign_order(si[0], sj[0]), sign_order(di[0], dj[0]))
                    or violates_no_cross(sign_order(si[1], sj[1]), sign_order(di[1], dj[1]))
                ):
                    V_cross.append((qi, qj, P_t, P_t1, t))

        # Stop early: downstream P^{>t} is undefined once a round breaks.
        if V_coll or V_cross:
            return {"V_coll": V_coll, "V_cross": V_cross, "fatal": fatal}

        layout = P_t1

    return {"V_coll": V_coll, "V_cross": V_cross, "fatal": fatal}


# ----------------------------
# Main verifier
# ----------------------------

def verify_output(cz_stages: List[List[Pair]], result: Dict[str, Any], *, require_exact_cz_match: bool = True) -> None:
    """End-to-end check of a compiled schedule.

    Walks the interleaved AOD / CZ items in order. A leading AOD block
    and back-to-back CZ stages (no AOD between them) are both allowed.
    Raises on the first violation.
    """
    if not isinstance(result, dict):
        raise ValueError("result must be a dict.")
    if "initial_layout" not in result:
        raise ValueError("result missing 'initial_layout'.")
    if "schedule" not in result:
        raise ValueError("result missing 'schedule'.")

    hw = Hardware.from_cz_stages(cz_stages)
    init = parse_layout(result["initial_layout"])
    schedule = result["schedule"]
    if not isinstance(schedule, list):
        raise ValueError("'schedule' must be a list.")

    # Initial layout checks (qubit set, bounds, ≤1 per trap)
    check_initial_layout(hw, init)

    layout = init
    stage_idx = 0
    i = 0
    while i < len(schedule):
        item = schedule[i]

        if isinstance(item, dict) and "aod_move_groups" in item:
            aod_groups = parse_move_groups(item)
            for r, rnd in enumerate(aod_groups, start=1):
                link = f"AOD (Seq {i})"
                check_aod_round(hw, layout, rnd, link_name=link)
                layout = apply_aod_round(layout, rnd)
            i += 1

        elif isinstance(item, dict) and "cz_stage" in item:
            cz_out = parse_cz_stage(item)
            
            if require_exact_cz_match:
                if stage_idx >= len(cz_stages):
                     raise ValueError("Schedule contains more CZ stages than input.")
                expected = stage_pair_set(cz_stages[stage_idx])
                got = stage_pair_set(cz_out)
                if got != expected:
                    missing = sorted(expected - got)
                    extra = sorted(got - expected)
                    raise ValueError(
                        f"[CZ{stage_idx + 1}] pair set mismatch. "
                        f"missing={missing}, extra={extra}"
                    )

            audit_groups_pre_cz(hw, layout, cz_out, stage_idx=stage_idx + 1)
            stage_idx += 1
            i += 1

        else:
            raise ValueError(f"Unknown schedule item at index {i}: {item.keys()}")

    if require_exact_cz_match and stage_idx < len(cz_stages):
        raise ValueError(f"Schedule missing CZ stages. Completed {stage_idx}/{len(cz_stages)}.")

    if require_exact_cz_match and stage_idx != len(cz_stages):
        raise ValueError(f"Schedule has {stage_idx} CZ stages but expected {len(cz_stages)}.")
    if stage_idx == 0:
        raise ValueError("Schedule contains no CZ stages.")

    if set(layout.keys()) != set(range(hw.n_qubits)):
        raise ValueError("Final layout qubit set mismatch.")
    for q, c in layout.items():
        if not hw.in_bounds(c):
            raise ValueError(f"Final layout out of bounds: q{q}@{c}")

