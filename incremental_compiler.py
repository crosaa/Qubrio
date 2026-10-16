"""Multi-agent compilation loop.

``compile_incremental`` drives the placement, routing, and optimization
agents stage by stage. Each agent is gated by a deterministic verifier
(or the hardware simulator, for the optimizer).
"""
from __future__ import annotations
import ast
import re
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from math import ceil, sqrt

CHECKPOINT_VERSION = 1

from api_client import DeepSeekClient
from prompt_initial_layout import INITIAL_LAYOUT_PROMPT_TEMPLATE
from prompt_target_layout import TARGET_LAYOUT_PROMPT_TEMPLATE
from prompt_transition import TRANSITION_PROMPT_TEMPLATE
from prompt_optimize import OPTIMIZE_PROMPT_TEMPLATE
from verify import (
    Hardware,
    infer_n_qubits,
    check_initial_layout,
    check_aod_round,
    apply_aod_round,
    diagnose_layout,
    diagnose_routing,
    parse_layout,
)
from performance_simulation import calculate_transition_performance

Coord = Tuple[int, int]
Pair = Tuple[int, int]
CZStages = List[List[Pair]]
Move = Tuple[int, Coord, Coord]
AODRound = List[Move]

# ── Thread-local LLM logger ──────────────────────────────────────────────────
_llm_log_local = threading.local()

def _set_llm_log_path(path: Optional[Path]):
    """Set the LLM log file path for the current thread."""
    _llm_log_local.path = path
    _llm_log_local.call_count = 0

def _llm_log(section: str, content: str):
    """Append a section to the thread-local LLM log file."""
    path = getattr(_llm_log_local, 'path', None)
    if path is None:
        return
    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"  {section}\n")
            f.write(f"{'='*80}\n")
            f.write(content)
            f.write("\n")
    except Exception:
        pass

def _llm_log_call(call_label: str, messages: list, content: str, reasoning: str):
    """Log a full LLM call: prompt, reasoning, and output."""
    path = getattr(_llm_log_local, 'path', None)
    if path is None:
        return
    _llm_log_local.call_count = getattr(_llm_log_local, 'call_count', 0) + 1
    n = _llm_log_local.call_count

    # Log prompt
    prompt_text = "\n".join(
        f"[{m['role']}]:\n{m['content']}" for m in messages
    )
    _llm_log(f"CALL #{n} — {call_label} — PROMPT", prompt_text)

    # Log reasoning (thinking)
    if reasoning:
        _llm_log(f"CALL #{n} — {call_label} — REASONING (THINKING)", reasoning)

    # Log output
    _llm_log(f"CALL #{n} — {call_label} — OUTPUT", content)


def _log_token_usage(call_label: str, usage: dict):
    """Append token usage to a machine-parseable JSONL file."""
    path = getattr(_llm_log_local, 'path', None)
    if path is None:
        return
    usage_path = path.parent / "token_usage.jsonl"
    n = getattr(_llm_log_local, 'call_count', 0)
    entry = {"call_number": n, "call_label": call_label, **usage}
    try:
        with open(usage_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def _write_checkpoint(checkpoint_path: Path, state: dict) -> None:
    """Atomic write via temp + rename so interrupts can't corrupt the file."""
    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp_path.replace(checkpoint_path)
    except Exception as exc:
        print(f"[CHECKPOINT WARN] failed to write {checkpoint_path}: {exc}")


def _layout_to_json(layout: Dict[int, Coord]) -> Dict[str, List[int]]:
    """Convert a Dict[int, (int,int)] layout to JSON-serializable form."""
    return {str(q): [int(pos[0]), int(pos[1])] for q, pos in sorted(layout.items())}


def format_cz_stage(stage: List[Pair]) -> str:
    if not stage:
        return "[]"
    pairs = ", ".join([f"({a}, {b})" for a, b in stage])
    return f"[{pairs}]"


def format_layout(layout: Dict[int, Coord]) -> str:
    lines = []
    for q in sorted(layout.keys()):
        x, y = layout[q]
        lines.append(f"  {q}: ({x}, {y})")
    return "{\n" + ",\n".join(lines) + "\n}"


def extract_python_dict(response: str) -> str:
    code_block_pattern = r"```(?:python)?\s*(.*?)\s*```"
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    
    if matches:
        response = matches[0]
    
    dict_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    matches = re.findall(dict_pattern, response, re.DOTALL)
    
    if not matches:
        raise ValueError("No Python dict found in LLM response")
    
    return max(matches, key=len)


def generate_initial_layout(
    cz_stages: CZStages,
    client: DeepSeekClient,
    max_retries: int = 10,
    verbose: bool = True
) -> Dict[int, Coord]:
    n_qubits = infer_n_qubits(cz_stages)
    side = ceil(sqrt(n_qubits))
    width = 2 * side
    height = side
    hw = Hardware(n_qubits=n_qubits, width=width, height=height)
    
    cz_stage1 = cz_stages[0]
    cz_stage1_str = format_cz_stage(cz_stage1)
    
    prompt = INITIAL_LAYOUT_PROMPT_TEMPLATE.format(
        width=width,
        height=height,
        n_qubits=n_qubits,
        cz_stage1=cz_stage1_str,
        n_qubits_minus_1=n_qubits - 1
    )
    
    last_response_text = None
    error_feedback = None
    
    for attempt in range(max_retries):
        if verbose:
            print(f"\n[Initial Layout CZ Stage 1] Attempt {attempt + 1}/{max_retries}")
        
        try:
            if last_response_text and error_feedback:
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": last_response_text},
                    {"role": "user", "content": f"The previous output was invalid.\n\nError Reason:\n{error_feedback}\n\nPlease fix the error and regenerate the layout."}
                ]
            else:
                messages = [{"role": "user", "content": prompt}]
            
            response, reasoning = client.call_with_reasoning(messages, temperature=0.7)
            last_response_text = response
            _label = f"Initial Layout Attempt {attempt+1}"
            _llm_log_call(_label, messages, response, reasoning)
            _log_token_usage(_label, client.get_normalized_usage())

            if verbose:
                print(f"[API Response] {len(response)} chars")
            
            dict_str = extract_python_dict(response)
            result = ast.literal_eval(dict_str)
            
            if "initial_layout" not in result:
                raise ValueError("Response missing 'initial_layout' key")
            
            layout = parse_layout(result["initial_layout"])
            check_initial_layout(hw, layout)
            # audit_groups_pre_cz(hw, layout, cz_stage1, stage_idx=1) # Removed: Initial layout no longer requires CZ1 pairing
            
            if verbose:
                print(f"[OK] Valid Initial layout generated")
            
            return layout
            
        except (ValueError, SyntaxError, KeyError) as e:
            error_feedback = f"Parsing/Verification Error: {e}"
            if verbose:
                print(f"[FAIL] Attempt {attempt + 1} failed: {e}")
            
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to generate initial layout after {max_retries} attempts: {e}")
    
    raise RuntimeError("Unexpected error in generate_initial_layout")


def _format_routing_feedback(diagnosis: Dict[str, Any]) -> Optional[str]:
    """Render ``diagnose_routing`` output as natural-language feedback.

    Returns ``None`` if the AOD sequence is fully valid. Fatal errors
    (unknown qubit, out-of-bounds, etc.) are surfaced before the
    categorical V_coll / V_cross sets.
    """
    fatal = diagnosis.get("fatal", [])
    V_coll = diagnosis.get("V_coll", [])
    V_cross = diagnosis.get("V_cross", [])

    if not (fatal or V_coll or V_cross):
        return None

    if fatal:
        msg = "# Routing Errors (must fix before category feedback)\n"
        for f in fatal:
            msg += f"  - {f}\n"
        return msg

    msg = (
        "# Routing Verifier Feedback\n\n"
        "Each violation is localized to a single AOD round t (0-indexed) and\n"
        "the offending qubit pair, so you can edit the affected segments\n"
        "while keeping the rest of the schedule.\n"
    )

    if V_coll:
        msg += (
            "\n## Collision set V_coll  "
            "(more than 1 atom occupies the same trap after action a_t)\n"
        )
        for q1, q2, P_t, P_t1, t in V_coll:
            c = P_t1.get(q1)
            msg += (
                f"  - step t={t}: q{q1} and q{q2} both end at {c} "
                f"(q{q1}: {P_t.get(q1)}→{P_t1.get(q1)}, "
                f"q{q2}: {P_t.get(q2)}→{P_t1.get(q2)})\n"
            )
        msg += (
            "  Fix: route one of the pair to a different trap, "
            "or split the conflicting moves into separate rounds. "
            "vacate-then-occupy is allowed within a single round.\n"
        )

    if V_cross:
        msg += (
            "\n## Crossing set V_cross  "
            "(strict x or y ordering of two movers flips within round t)\n"
        )
        for q1, q2, P_t, P_t1, t in V_cross:
            msg += (
                f"  - step t={t}: q{q1}: {P_t.get(q1)}→{P_t1.get(q1)}, "
                f"q{q2}: {P_t.get(q2)}→{P_t1.get(q2)}\n"
            )
        msg += (
            "  Fix: move q_i and q_j in DIFFERENT rounds, "
            "or re-route via an intermediate vacant trap so their "
            "relative x and y orders are preserved.\n"
        )

    msg += (
        "\nAn early step index t suggests the departure strategy is flawed; "
        "a late t suggests only the final approach to the target needs "
        "adjustment. Modify the offending segments and keep the valid "
        "earlier rounds unchanged."
    )
    return msg


def auto_resolve_crossings(
    hw: Hardware,
    current_layout: Dict[int, Coord],
    aod_groups: List[AODRound],
) -> Optional[List[AODRound]]:
    """Peel no-crossing offenders into separate rounds.

    Returns the patched schedule, or ``None`` if a round expands past 25
    sub-rounds or hits a non-crossing error (caller falls back to LLM
    feedback in that case).
    """
    resolved_groups: List[AODRound] = []
    layout = dict(current_layout)

    for rnd in aod_groups:
        sub_rounds: List[AODRound] = [rnd]
        idx = 0
        while idx < len(sub_rounds):
            if len(sub_rounds) > 25:
                return None

            curr_rnd = sub_rounds[idx]
            try:
                check_aod_round(hw, layout, curr_rnd, link_name="AutoSplit")
                layout = apply_aod_round(layout, curr_rnd)
                resolved_groups.append(curr_rnd)
                idx += 1
            except ValueError as e:
                err_msg = str(e)
                if "symbolic no-crossing violated" not in err_msg:
                    return None
                match = re.search(r"moving pair \(q(\d+),q(\d+)\)", err_msg)
                if not match:
                    return None
                qi, qj = int(match.group(1)), int(match.group(2))
                idx_i = next((i for i, m in enumerate(curr_rnd) if m[0] == qi), -1)
                idx_j = next((i for i, m in enumerate(curr_rnd) if m[0] == qj), -1)
                later_q = qi if idx_i > idx_j else qj
                new_curr = [m for m in curr_rnd if m[0] != later_q]
                new_next = [m for m in curr_rnd if m[0] == later_q]
                sub_rounds[idx] = new_curr
                sub_rounds.insert(idx + 1, new_next)

    return resolved_groups


def _accept_candidate(new_m: Dict[str, float], best_m: Dict[str, float]) -> bool:
    """Pareto rule: improve T_exe or fidelity without degrading the other."""
    new_t = new_m["total_duration"]
    new_f = new_m["final_fidelity"]
    best_t = best_m["total_duration"]
    best_f = best_m["final_fidelity"]

    t_better = new_t < best_t
    f_better = new_f > best_f
    t_worse  = new_t > best_t
    f_worse  = new_f < best_f

    return (t_better and not f_worse) or (f_better and not t_worse)


def _format_optimize_feedback(
    prev_metrics: Dict[str, float],
    best_metrics: Dict[str, float],
    rejection_log: List[Dict[str, Any]],
) -> str:
    """Two-layer feedback: outcome (T_exe, F) plus diagnostic
    (transfer / coherence) breakdown, with up to three recent
    Pareto-rejected candidates appended as negative constraints.
    """
    msg = (
        "# Iteration Feedback\n\n"
        "## Your previous attempt\n"
        "**Outcome metrics:**\n"
        f"- T_exe    = {prev_metrics['total_duration']:.1f} us\n"
        f"- Fidelity = {prev_metrics['final_fidelity']:.4f}\n\n"
        "**Diagnostic metrics (decomposed fidelity sources):**\n"
        f"- Transfer fidelity   = {prev_metrics['movement_fidelity']:.4f}   "
        "(low ⇒ too many SLM↔AOD pickups)\n"
        f"- Coherence fidelity  = {prev_metrics['coherence_fidelity']:.4f}  "
        "(low ⇒ schedule too long, decoherence dominates)\n\n"
        "## Best schedule so far\n"
        f"- T_exe = {best_metrics['total_duration']:.1f} us, "
        f"Fidelity = {best_metrics['final_fidelity']:.4f}\n"
        f"  (transfer = {best_metrics['movement_fidelity']:.4f}, "
        f"coherence = {best_metrics['coherence_fidelity']:.4f})\n"
    )

    rejected = [r for r in rejection_log if r.get("type") == "rejected"]
    if rejected:
        msg += "\n## Previously rejected candidates (avoid these regions)\n"
        for r in rejected[-3:]:  # last 3 to avoid context bloat
            m = r["metrics"]
            msg += (
                f"- T={m['total_duration']:.0f}us, F={m['final_fidelity']:.3f} "
                f"(transfer={m['movement_fidelity']:.3f}, "
                f"coherence={m['coherence_fidelity']:.3f})\n"
            )

    msg += (
        "\n## How to use the diagnostic\n"
        "- If transfer fidelity is the bottleneck ⇒ reduce SLM↔AOD pickups by merging moves into chains\n"
        "- If coherence fidelity is the bottleneck ⇒ shorten total schedule duration by folding orthogonal steps\n\n"
        "## Acceptance Rule\n"
        "A new schedule is accepted only if it improves at least one of T_exe / Fidelity\n"
        "without degrading the other. Generate a new schedule that beats the current best."
    )
    return msg


def _format_invalid_feedback(
    error_msg: str,
    best_metrics: Dict[str, float],
) -> str:
    """Feedback for an optimizer attempt that failed physical verification."""
    return (
        "# Iteration Feedback — Invalid Candidate\n\n"
        "Your previous schedule did not pass physical verification and was "
        "rejected before simulation. The verifier reports:\n\n"
        f"{error_msg}\n\n"
        "## Best schedule so far\n"
        f"- T_exe = {best_metrics['total_duration']:.1f} us, "
        f"Fidelity = {best_metrics['final_fidelity']:.4f}\n\n"
        "## Acceptance Rule\n"
        "Generate a new schedule that (1) satisfies all hard physical "
        "constraints (no-crossing on x and y axes, ≤1 atom per trap at end "
        "of each round, unique destinations within a round) and "
        "(2) improves at least one of T_exe / Fidelity over the best so far "
        "without degrading the other."
    )


def _build_optimize_messages(
    base_prompt: str,
    prev_schedule_text: Optional[str],
    prev_metrics: Optional[Dict[str, float]],
    prev_invalid_error: Optional[str],
    best_metrics: Dict[str, float],
    rejection_log: List[Dict[str, Any]],
    attempt: int,
) -> list:
    """Build the optimizer's message list for one attempt.

    First attempt is just the base prompt. Subsequent attempts add the
    LLM's previous output as the assistant turn plus a user turn of
    feedback. The feedback variant depends on whether the previous
    candidate was valid (outcome + diagnostic), invalid (verifier
    error), or absent (generic schema reminder).
    """
    if attempt == 1:
        return [{"role": "user", "content": base_prompt}]

    messages = [{"role": "user", "content": base_prompt}]

    if prev_schedule_text is not None and prev_metrics is not None:
        messages.append({"role": "assistant", "content": prev_schedule_text})
        feedback = _format_optimize_feedback(prev_metrics, best_metrics, rejection_log)
        messages.append({"role": "user", "content": feedback})
    elif prev_schedule_text is not None and prev_invalid_error is not None:
        messages.append({"role": "assistant", "content": prev_schedule_text})
        messages.append({
            "role": "user",
            "content": _format_invalid_feedback(prev_invalid_error, best_metrics),
        })
    else:
        messages.append({
            "role": "user",
            "content": (
                "# Iteration Feedback\n\n"
                "Your previous attempt failed parsing or verification. "
                "Please re-attempt and ensure the output strictly follows the JSON schema."
            ),
        })

    return messages


def optimize_aod_moves(
    aod_groups: List[AODRound],
    current_layout: Dict[int, Coord],
    target_layout: Dict[int, Coord],
    stage_idx: int,
    hw: Hardware,
    client: DeepSeekClient,
    max_retries: int = 5,
    verbose: bool = True,
) -> List[AODRound]:
    """Refine an already-valid AOD schedule via simulator-guided feedback.

    Each attempt after the first sees the previous schedule's
    outcome + diagnostic metrics (or its verifier error, if invalid)
    and recently rejected candidates as negative constraints.
    """
    # ----- format the unoptimized schedule and layout context -----
    formatted_groups = []
    for round_moves in aod_groups:
        formatted_round = []
        for move in round_moves:
            q, (x1, y1), (x2, y2) = move
            formatted_round.append([q, [x1, y1], [x2, y2]])
        formatted_groups.append(formatted_round)

    unoptimized_json = json.dumps(formatted_groups)

    current_layout_str = "\n".join([f"q{q}: {coord}" for q, coord in sorted(current_layout.items())])
    target_layout_str = "\n".join([f"q{q}: {coord}" for q, coord in sorted(target_layout.items())])

    occupied_coords = set(current_layout.values())
    vacant_coords = []
    for y in range(hw.height):
        for x in range(hw.width):
            if (x, y) not in occupied_coords:
                vacant_coords.append((x, y))
    vacant_traps_str = "\n".join([str(c) for c in vacant_coords])

    base_prompt = OPTIMIZE_PROMPT_TEMPLATE.format(
        width=hw.width,
        height=hw.height,
        n_qubits=hw.n_qubits,
        current_layout=current_layout_str,
        vacant_traps=vacant_traps_str,
        target_layout=target_layout_str,
        unoptimized_aod_groups=unoptimized_json,
    )

    if verbose:
        print(
            f"\n[OPTIMIZE] Sending {len(aod_groups)} rounds to optimizer "
            f"for {max_retries} attempts."
        )

    def compute_metrics(schedule: List[AODRound]) -> Dict[str, float]:
        return calculate_transition_performance(schedule, hw.n_qubits)

    # ----- iterative refinement state -----
    best_schedule = aod_groups
    best_metrics  = compute_metrics(aod_groups)

    prev_schedule_text: Optional[str] = None  # raw LLM output to replay as assistant turn
    prev_metrics: Optional[Dict[str, float]] = None
    prev_invalid_error: Optional[str] = None  # verifier error from previous (invalid) attempt
    rejection_log: List[Dict[str, Any]] = []  # negative-constraint cache for the next attempt

    for attempt in range(1, max_retries + 1):
        messages = _build_optimize_messages(
            base_prompt=base_prompt,
            prev_schedule_text=prev_schedule_text,
            prev_metrics=prev_metrics,
            prev_invalid_error=prev_invalid_error,
            best_metrics=best_metrics,
            rejection_log=rejection_log,
            attempt=attempt,
        )

        try:
            res, reasoning = client.call_with_reasoning(messages, temperature=0.7)
            _opt_label = f"Optimize CZ{stage_idx+1}→CZ{stage_idx+2} Attempt {attempt}"
            _llm_log_call(_opt_label, messages, res, reasoning)
            _log_token_usage(_opt_label, client.get_normalized_usage())

            if verbose:
                print(f"[API Response] {len(res)} chars (Attempt {attempt}/{max_retries})")

            # ----- parse JSON out of the response -----
            code_block_pattern = r"```(?:json)?\s*(\[.*?\])\s*```"
            matches = re.findall(code_block_pattern, res, re.DOTALL)
            if matches:
                res_str = matches[0]
            else:
                start = res.find('[')
                end = res.rfind(']')
                if start != -1 and end != -1:
                    res_str = res[start:end+1]
                else:
                    raise ValueError("Could not extract JSON from response")

            try:
                optimized_raw = json.loads(res_str)
            except json.JSONDecodeError:
                # LLM sometimes drops the trailing ']' on long arrays.
                fixed = res_str
                open_sq = fixed.count('[')
                close_sq = fixed.count(']')
                if open_sq > close_sq:
                    fixed = fixed + (']' * (open_sq - close_sq))
                optimized_raw = json.loads(fixed)
                if verbose:
                    print(f"  -> [AUTO-FIX] Added {open_sq - close_sq} missing ']' to LLM output")

            optimized_aod_groups = []
            for rnd in optimized_raw:
                moves = []
                for m in rnd:
                    src = (int(m[1][0]), int(m[1][1]))
                    dst = (int(m[2][0]), int(m[2][1]))
                    moves.append((int(m[0]), src, dst))
                optimized_aod_groups.append(moves)

            # ----- verify physical validity -----
            layout = dict(current_layout)
            for r, rnd in enumerate(optimized_aod_groups, start=1):
                link = f"[CZ{stage_idx+1} → CZ{stage_idx+2}] OPTIMIZED AOD round {r}"
                check_aod_round(hw, layout, rnd, link_name=link)
                layout = apply_aod_round(layout, rnd)

            for q, pos in target_layout.items():
                if layout.get(q) != pos:
                    raise ValueError(
                        f"Routing incomplete! Qubit {q} ended at {layout.get(q)}, "
                        f"but must end at {pos}."
                    )

            # ----- evaluate against simulator -----
            cand_metrics = compute_metrics(optimized_aod_groups)
            if verbose:
                print(
                    f"  -> Valid! Rounds: {len(optimized_aod_groups)}, "
                    f"T={cand_metrics['total_duration']:.1f}us, "
                    f"F={cand_metrics['final_fidelity']:.5f} "
                    f"(transfer={cand_metrics['movement_fidelity']:.4f}, "
                    f"coherence={cand_metrics['coherence_fidelity']:.4f})"
                )

            if _accept_candidate(cand_metrics, best_metrics):
                best_schedule = optimized_aod_groups
                best_metrics = cand_metrics
                if verbose:
                    print(f"  -> [ACCEPTED] new best schedule")
            else:
                rejection_log.append({"type": "rejected", "metrics": cand_metrics})
                if verbose:
                    print(f"  -> Rejected (no Pareto improvement over current best)")

            # carry forward for next iteration's feedback
            prev_schedule_text = res
            prev_metrics = cand_metrics
            prev_invalid_error = None

        except (ValueError, SyntaxError, KeyError, json.JSONDecodeError) as e:
            # UnicodeError subclasses ValueError; let console-encoding
            # glitches escape rather than be relabeled as bad LLM output.
            if isinstance(e, UnicodeError):
                raise
            error_msg = str(e)
            if verbose:
                print(f"  -> [OPTIMIZE FAILED] Attempt {attempt}/{max_retries}. Error: {error_msg}")
            rejection_log.append({"type": "invalid", "reason": error_msg})
            # Keep `res` as the next assistant turn paired with the verifier
            # error so the model sees what it produced and why it was bad.
            prev_schedule_text = res
            prev_metrics = None
            prev_invalid_error = error_msg

    if verbose:
        print(
            f"[OPTIMIZE FINISHED] Best schedule rounds: {len(best_schedule)} "
            f"(from original {len(aod_groups)}); "
            f"final T={best_metrics['total_duration']:.1f}us "
            f"F={best_metrics['final_fidelity']:.4f}"
        )

    return best_schedule

def _format_placement_feedback(diagnosis: Dict[str, Any]) -> Optional[str]:
    """Two-stage feedback for the Placement Agent.

    Surfaces V_coll alone first (resolve all collisions before mixing
    constraints). Once V_coll is empty, V_adj and V_xtalk are reported
    jointly because they apply opposing spatial pressures. Returns None
    when the layout is valid; fatal errors are surfaced ahead of any
    categorical feedback.
    """
    fatal = diagnosis.get("fatal", [])
    V_coll = diagnosis.get("V_coll", [])
    V_adj = diagnosis.get("V_adj", [])
    V_xtalk = diagnosis.get("V_xtalk", [])

    if not (fatal or V_coll or V_adj or V_xtalk):
        return None

    if fatal:
        msg = "# Layout Errors (must fix before category feedback)\n"
        for f in fatal:
            msg += f"  - {f}\n"
        return msg

    if V_coll:
        # Stage 1: collisions only.
        msg = (
            "# Stage 1 Feedback — Resolve Collisions Only\n\n"
            "The following qubit pairs are colliding (assigned to overlapping "
            "trap positions). Adjacency (V_adj) and crosstalk (V_xtalk) "
            "feedback will be given AFTER all collisions are resolved.\n\n"
            "## Collision set V_coll  (qi, qj, P(qi), P(qj))\n"
        )
        for q1, q2, c1, c2 in V_coll:
            msg += f"  - (q{q1}, q{q2}, {c1}, {c2})\n"
        msg += (
            "\nFix: assign each colliding qubit to a distinct trap. "
            "Two qubits may share a group only if they form a CZ pair this stage; "
            "even then they must occupy different (x, y) coordinates "
            "(one at even x = LEFT trap, one at odd x = RIGHT trap)."
        )
        return msg

    # Stage 2: joint V_adj + V_xtalk reporting.
    msg = (
        "# Stage 2 Feedback — Resolve Adjacency and Crosstalk Jointly\n\n"
        "Collisions are resolved. The remaining violations impose OPPOSING "
        "spatial pressures and must be balanced together, not in isolation.\n"
    )
    if V_adj:
        msg += "\n## Adjacency set V_adj  (CZ pairs placed too far apart)  (qi, qj, P(qi), P(qj))\n"
        for q1, q2, c1, c2 in V_adj:
            msg += f"  - (q{q1}, q{q2}, {c1}, {c2})\n"
        msg += "  Fix: bring these pairs into the SAME group (same x//2 and same y).\n"
    if V_xtalk:
        msg += "\n## Crosstalk set V_xtalk  (non-CZ pairs placed in the same group)  (qi, qj, P(qi), P(qj))\n"
        for q1, q2, c1, c2 in V_xtalk:
            msg += f"  - (q{q1}, q{q2}, {c1}, {c2})\n"
        msg += "  Fix: separate these pairs into DIFFERENT groups.\n"
    msg += (
        "\nDo not over-correct in one direction. Moving a qubit to satisfy V_adj "
        "may create a new V_xtalk (or vice versa). Plan placements that satisfy "
        "both sets simultaneously."
    )
    return msg


def generate_target_layout(
    stage_idx: int,
    current_layout: Dict[int, Coord],
    cz_stages: CZStages,
    hw: Hardware,
    client: DeepSeekClient,
    max_retries: int = 10,
    verbose: bool = True,
) -> Dict[int, Coord]:
    cz_stage_next = cz_stages[stage_idx + 1] if stage_idx + 1 < len(cz_stages) else []
    cz_stage_future_1 = cz_stages[stage_idx + 2] if stage_idx + 2 < len(cz_stages) else []
    cz_stage_future_2 = cz_stages[stage_idx + 3] if stage_idx + 3 < len(cz_stages) else []

    future_stages_str = ""
    if cz_stage_future_1:
        future_stages_str += f"- CZ Stage {stage_idx + 3}: {format_cz_stage(cz_stage_future_1)}\n"
    if cz_stage_future_2:
        future_stages_str += f"- CZ Stage {stage_idx + 4}: {format_cz_stage(cz_stage_future_2)}\n"
    if not future_stages_str:
        future_stages_str = "None\n"

    prompt = TARGET_LAYOUT_PROMPT_TEMPLATE.format(
        width=hw.width,
        height=hw.height,
        n_qubits=hw.n_qubits,
        n_qubits_minus_1=hw.n_qubits - 1,
        stage_from=stage_idx + 1,
        stage_to=stage_idx + 2,
        current_layout=format_layout(current_layout),
        cz_stage_next=format_cz_stage(cz_stage_next),
        future_stages=future_stages_str.strip()
    )

    last_response_text = None
    error_feedback = None
    failure_history = []

    for attempt in range(max_retries):
        if verbose:
            print(f"\n[Target Layout CZ Stage {stage_idx+2}] Attempt {attempt + 1}/{max_retries}")

        try:
            if last_response_text and error_feedback:
                 history_msg = ""
                 if failure_history:
                     history_msg = "\n\n# History of Failed Attempts:\n"
                     for i, code, err in failure_history:
                          code_excerpt = code if len(code) <= 2000 else code[:2000] + "\n... [truncated]"
                          history_msg += f"--- Attempt {i} ---\nOutput:\n{code_excerpt}\nError:\n{err}\n"

                 messages = [
                     {"role": "user", "content": prompt},
                     {"role": "assistant", "content": last_response_text},
                     {"role": "user", "content": f"{history_msg}\n\n========================================\nThe previous output was invalid.\n\nError Reason:\n{error_feedback}\n\nPlease fix the error."}
                 ]
            else:
                 messages = [{"role": "user", "content": prompt}]
            
            response, reasoning = client.call_with_reasoning(messages, temperature=0.7)
            last_response_text = response
            _tl_label = f"Target Layout CZ{stage_idx+1}→CZ{stage_idx+2} Attempt {attempt+1}"
            _llm_log_call(_tl_label, messages, response, reasoning)
            _log_token_usage(_tl_label, client.get_normalized_usage())

            # ---- Parse the LLM output ----
            try:
                dict_str = extract_python_dict(response)
                result = ast.literal_eval(dict_str)
                if "target_layout" not in result:
                    raise ValueError("Missing 'target_layout' key in response")
                layout_raw = result["target_layout"]
                target_layout = {int(k): (int(v[0]), int(v[1])) for k, v in layout_raw.items()}
            except Exception as parse_err:
                error_feedback = f"Parsing error: {parse_err}"
                if verbose:
                    print(f"[FAIL] {error_feedback}")
                failure_history.append((attempt + 1, response, error_feedback))
                continue

            diagnosis = diagnose_layout(hw, target_layout, cz_stage_next)
            stage_feedback = _format_placement_feedback(diagnosis)

            if stage_feedback is None:
                if verbose:
                    print(f"[OK] Valid Target Layout generated!")
                return target_layout

            error_feedback = stage_feedback
            n_coll = len(diagnosis["V_coll"])
            n_adj = len(diagnosis["V_adj"])
            n_xtalk = len(diagnosis["V_xtalk"])
            n_fatal = len(diagnosis["fatal"])
            if verbose:
                if n_fatal:
                    print(f"[FAIL] {n_fatal} fatal error(s)")
                elif n_coll:
                    print(f"[FAIL] Stage 1: {n_coll} collision(s) — surfacing V_coll only")
                else:
                    print(f"[FAIL] Stage 2: {n_adj} adjacency, {n_xtalk} crosstalk — joint feedback")
            short_err = (
                f"V_coll={n_coll}, V_adj={n_adj}, V_xtalk={n_xtalk}, fatal={n_fatal}"
            )
            failure_history.append((attempt + 1, last_response_text or "", short_err))

        except Exception as e:
            if isinstance(e, UnicodeError):
                raise
            error_feedback = str(e)
            if verbose:
                print(f"[FAIL] {error_feedback}")
            failure_history.append((attempt + 1, last_response_text or "", error_feedback))

    raise RuntimeError(f"Failed to generate valid target layout after {max_retries} attempts.")

def generate_transition(
    stage_idx: int,
    current_layout: Dict[int, Coord],
    cz_stages: CZStages,
    hw: Hardware,
    client: DeepSeekClient,
    max_retries: int = 10,
    verbose: bool = True,
    external_error_feedback: str = "",
    previous_failures: List[Tuple[int, str, str]] = None,  # (retry_idx, code, error) from OUTER loop
) -> Tuple[List[AODRound], Dict[int, Coord]]:
    if previous_failures is None:
        previous_failures = []

    try:
        target_layout = generate_target_layout(
            stage_idx=stage_idx,
            current_layout=current_layout,
            cz_stages=cz_stages,
            hw=hw,
            client=client,
            max_retries=max_retries,
            verbose=verbose
        )
    except Exception as e:
        raise RuntimeError(f"Target Layout failed: {e}")

    prompt = TRANSITION_PROMPT_TEMPLATE.format(
        width=hw.width,
        height=hw.height,
        current_layout=format_layout(current_layout),
        target_layout=format_layout(target_layout)
    )
    
    last_response_text = None
    error_feedback = None
    failure_history: List[Tuple[int, str, str]] = []  # (attempt_idx, code, error)
    
    for attempt in range(max_retries):
        if verbose:
            print(f"\n[Transition to CZ Stage {stage_idx+2}] Attempt {attempt + 1}/{max_retries}")
        
        try:
            if last_response_text and error_feedback:
                # Replay this inner loop's failures so the LLM can see why
                # each previous attempt was rejected.
                history_msg = ""
                if failure_history:
                    history_msg = "\n\n# History of Previous Failed Attempts in this Loop:\n"
                    for i, code, err in failure_history:
                        code_excerpt = code if len(code) <= 2000 else code[:2000] + "\n... [truncated]"
                        history_msg += f"--- Attempt {i} ---\nOutput:\n{code_excerpt}\nError:\n{err}\n"

                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": last_response_text},
                    {"role": "user", "content": f"{history_msg}\n\n========================================\nThe previous output was invalid.\n\nError Reason:\n{error_feedback}\n\nPlease fix the error and regenerate the AOD moves."}
                ]
            else:
                # First attempt: append any failures from prior outer retries
                # so the LLM avoids re-proposing already-rejected schedules.
                base_prompt = prompt
                if previous_failures:
                    history_msg = "\n\n# PREVIOUS FAILED SOLUTIONS (DO NOT REPEAT):\n"
                    for i, code, err in previous_failures:
                        code_excerpt = code if len(code) <= 2000 else code[:2000] + "\n... [truncated]"
                        history_msg += f"--- Outer Retry {i+1} ---\nOutput:\n{code_excerpt}\nError:\n{err}\n"
                    base_prompt += history_msg

                messages = [{"role": "user", "content": base_prompt}]
            
            response, reasoning = client.call_with_reasoning(messages, temperature=0.7)
            last_response_text = response
            _tr_label = f"Transition CZ{stage_idx+1}→CZ{stage_idx+2} Inner Attempt {attempt+1}"
            _llm_log_call(_tr_label, messages, response, reasoning)
            _log_token_usage(_tr_label, client.get_normalized_usage())
            
            if verbose:
                print(f"[API Response] {len(response)} chars")
                print(f"[LLM RAW OUTPUT]")
                print("-" * 60)
                print(response)
                print("-" * 60)
            
            dict_str = extract_python_dict(response)
            result = ast.literal_eval(dict_str)
            
            if "aod_move_groups" not in result:
                raise ValueError("Response missing 'aod_move_groups' key")
            
            aod_groups = result["aod_move_groups"]
            
            if not isinstance(aod_groups, list):
                raise ValueError("'aod_move_groups' must be a list")
            
            formatted_groups = []
            for r, rnd in enumerate(aod_groups, start=1):
                if not isinstance(rnd, list):
                    raise ValueError(f"Round {r} must be a list")
                
                moves: AODRound = []
                for m in rnd:
                    if not (isinstance(m, (tuple, list)) and len(m) == 3):
                        raise ValueError(f"Move must be (q, src, dst), got {m}")
                    q, src, dst = m
                    if not (isinstance(src, (tuple, list)) and len(src) == 2):
                        raise ValueError(f"src must be (x,y), got {src}")
                    if not (isinstance(dst, (tuple, list)) and len(dst) == 2):
                        raise ValueError(f"dst must be (x,y), got {dst}")
                    moves.append((int(q), (int(src[0]), int(src[1])), (int(dst[0]), int(dst[1]))))
                formatted_groups.append(moves)

            # Peel local no-crossing offenders before falling back to LLM feedback.
            resolved_groups = auto_resolve_crossings(hw, current_layout, formatted_groups)
            aod_groups = resolved_groups if resolved_groups is not None else formatted_groups

            diagnosis = diagnose_routing(hw, current_layout, aod_groups)
            routing_feedback = _format_routing_feedback(diagnosis)
            if routing_feedback is not None:
                raise ValueError(routing_feedback)

            # Replay rounds to obtain the final layout and check it matches the target.
            layout = dict(current_layout)
            for rnd in aod_groups:
                layout = apply_aod_round(layout, rnd)

            for q, pos in target_layout.items():
                if layout[q] != pos:
                    raise ValueError(
                        f"Routing incomplete! Qubit {q} ended at {layout.get(q)}, "
                        f"but must end at {pos} to satisfy Target Layout."
                    )

            if verbose:
                print("[VERIFY COMPLETE] Atoms successfully routed to Target Layout.")
                print(f"[OK] Transition generated ({len(aod_groups)} rounds)")
            
            # Keep a copy before optimization
            unoptimized_aod_groups = [list(rnd) for rnd in aod_groups]

            # Optimize aod_groups before returning
            aod_groups = optimize_aod_moves(
                aod_groups, current_layout, target_layout, stage_idx, hw, client, max_retries=5, verbose=verbose
            )

            return aod_groups, target_layout, unoptimized_aod_groups
            
        except (ValueError, SyntaxError, KeyError) as e:
            # UnicodeError subclasses ValueError; let console-encoding
            # glitches escape rather than be relabeled as a verifier failure.
            if isinstance(e, UnicodeError):
                raise
            error_feedback = str(e)
            
            # Record this failure
            if last_response_text:
                # Try to extract the code block for cleaner history, or use raw if extract fails
                try:
                    code_content = extract_python_dict(last_response_text)
                except Exception:
                    code_content = last_response_text[:500] + "... (parsing failed)"
                
                failure_history.append((attempt + 1, code_content, str(e)))

            if verbose:
                print(f"[FAIL] Attempt {attempt + 1} failed: {e}")
            
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to generate transition {stage_idx+1}→{stage_idx+2} after {max_retries} attempts: {e}")
    
    raise RuntimeError("Unexpected error in generate_transition")


def compile_incremental(
    cz_stages: CZStages,
    api_key: str,
    provider: str = "deepseek",
    inner_max_retries: int = 10,
    outer_max_retries: int = 5,
    verbose: bool = True,
    llm_log_path: Optional[Path] = None,
    model: Optional[str] = None,
    effort: str = "medium",
) -> Dict[str, Any]:
    _set_llm_log_path(llm_log_path)

    client = DeepSeekClient(api_key, provider=provider, model=model, effort=effort)
    
    n_qubits = infer_n_qubits(cz_stages)
    side = ceil(sqrt(n_qubits))
    hw = Hardware(n_qubits=n_qubits, width=2 * side, height=side)
    
    if verbose:
        print("=" * 60)
        print(f"Incremental Quantum Compiler")
        print(f"N_QUBITS={n_qubits}, WIDTH={hw.width}, HEIGHT={hw.height}")
        print(f"CZ Stages: {len(cz_stages)}")
        print(f"Inner max retries: {inner_max_retries}, Outer max retries: {outer_max_retries}")
        print("=" * 60)
    
    initial_layout = generate_initial_layout(cz_stages, client, inner_max_retries, verbose)

    schedule = []
    unoptimized_schedule = []
    current_layout = initial_layout

    # Early checkpoint: lets a resume pick up from the initial layout
    # without re-running that LLM call, even if stage 1 later fails.
    checkpoint_path = (llm_log_path.parent / "checkpoint.json") if llm_log_path else None
    if checkpoint_path is not None:
        _write_checkpoint(checkpoint_path, {
            "version": CHECKPOINT_VERSION,
            "n_qubits": n_qubits,
            "width": hw.width,
            "height": hw.height,
            "initial_layout": _layout_to_json(initial_layout),
            "current_layout": _layout_to_json(current_layout),
            "schedule": schedule,
            "unoptimized_schedule": unoptimized_schedule,
            "completed_stages": 0,
            "total_stages": len(cz_stages),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })

    for stage_idx, cz_stage in enumerate(cz_stages):
        previous_failures = []

        for retry_attempt in range(outer_max_retries):
            try:
                # ``generate_transition`` is parameterized by the PREVIOUS
                # stage index (it computes the move into stage_idx + 1), so
                # to target the current stage we pass ``stage_idx - 1``.
                # generate_transition takes the previous stage index (it
                # produces the layout for stage_idx + 1).
                aod_groups, target_layout, unopt_aod_groups = generate_transition(
                    stage_idx - 1,
                    current_layout,
                    cz_stages,
                    hw,
                    client,
                    inner_max_retries,
                    verbose,
                    previous_failures=previous_failures,
                )
                
                # Reshape AOD rounds into JSON-friendly lists
                # ([q, [x1, y1], [x2, y2]] per move).
                formatted_aod_groups = []
                for rnd in aod_groups:
                    formatted_rnd = []
                    for move in rnd:
                        q, (x1, y1), (x2, y2) = move
                        formatted_rnd.append([q, [x1, y1], [x2, y2]])
                    formatted_aod_groups.append(formatted_rnd)

                formatted_unopt_aod_groups = []
                for rnd in unopt_aod_groups:
                    formatted_rnd = []
                    for move in rnd:
                        q, (x1, y1), (x2, y2) = move
                        formatted_rnd.append([q, [x1, y1], [x2, y2]])
                    formatted_unopt_aod_groups.append(formatted_rnd)

                formatted_target_layout = {str(q): [pos[0], pos[1]] for q, pos in sorted(target_layout.items())}

                schedule.append({
                    "target_layout": formatted_target_layout,
                    "aod_move_groups": formatted_aod_groups
                })
                unoptimized_schedule.append({
                    "target_layout": formatted_target_layout,
                    "aod_move_groups": formatted_unopt_aod_groups
                })

                # CZ stage entry follows the AOD block in the schedule.
                schedule.append({"cz_stage": cz_stage})
                unoptimized_schedule.append({"cz_stage": cz_stage})

                for rnd in aod_groups:
                    current_layout = apply_aod_round(current_layout, rnd)

                if verbose:
                    print(f"[OK] Transition to CZ Stage {stage_idx + 1} completed")

                # Checkpoint after the stage is confirmed done — the only
                # point where schedule and current_layout are consistent.
                if checkpoint_path is not None:
                    _write_checkpoint(checkpoint_path, {
                        "version": CHECKPOINT_VERSION,
                        "n_qubits": n_qubits,
                        "width": hw.width,
                        "height": hw.height,
                        "initial_layout": _layout_to_json(initial_layout),
                        "current_layout": _layout_to_json(current_layout),
                        "schedule": schedule,
                        "unoptimized_schedule": unoptimized_schedule,
                        "completed_stages": stage_idx + 1,
                        "total_stages": len(cz_stages),
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                    })

                break
                
            except RuntimeError as e:
                error_msg = str(e)

                # We never see the inner loop's bad schedule here, so the
                # outer log records the error message with a placeholder.
                previous_failures.append((retry_attempt, "(Code from this attempt was discarded)", error_msg))

                if verbose:
                    print(f"[FAIL] Transition verification failed (attempt {retry_attempt + 1}/{outer_max_retries})")
                    print(f"[ERROR FEEDBACK] {error_msg}")

                # Roll back to the last confirmed stage (each contributes
                # an AOD entry then a CZ entry, hence stage_idx * 2).
                confirmed_len = stage_idx * 2
                del schedule[confirmed_len:]
                del unoptimized_schedule[confirmed_len:]

                # Rebuild current_layout by replaying confirmed AOD rounds.
                current_layout = dict(initial_layout)
                for i in range(0, confirmed_len, 2):
                    for rnd_formatted in schedule[i]["aod_move_groups"]:
                        rnd = [(m[0], tuple(m[1]), tuple(m[2])) for m in rnd_formatted]
                        current_layout = apply_aod_round(current_layout, rnd)
                
                if retry_attempt == outer_max_retries - 1:
                    raise RuntimeError(f"Failed to generate valid transition to CZ Stage {stage_idx + 1} after {outer_max_retries} outer attempts")

    
    # Format initial layout for final output
    formatted_initial_layout = {str(q): [pos[0], pos[1]] for q, pos in sorted(initial_layout.items())}

    result = {
        "n_qubits": n_qubits,
        "width": hw.width,
        "height": hw.height,
        "initial_layout": formatted_initial_layout,
        "schedule": schedule
    }

    unoptimized_result = {
        "n_qubits": n_qubits,
        "width": hw.width,
        "height": hw.height,
        "initial_layout": formatted_initial_layout,
        "schedule": unoptimized_schedule
    }

    if verbose:
        print("\n" + "=" * 60)
        print("[SUCCESS] Incremental compilation complete")
        print("=" * 60)

    return result, unoptimized_result


if __name__ == "__main__":
    print(
        "\n[INFO] incremental_compiler.py is a library module.\n"
        "To run the multi-threaded compiler across all datasets, use:\n\n"
        "    python run_all.py\n\n"
        "Options:\n"
        "    --workers N          Max parallel threads (default: one per dataset)\n"
        "    --inner-retries N    Inner retry limit per LLM call (default: 10)\n"
        "    --outer-retries N    Outer retry limit per transition (default: 5)\n"
        "    --data-dir PATH      Root data directory (default: data/)\n"
    )

