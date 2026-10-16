"""Recover compile state from an interrupted run.

``load_checkpoint`` is the preferred path (reads ``checkpoint.json``).
``parse_job_logs`` is a legacy regex-based fallback for older runs that
predate the checkpoint feature.
"""
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

Coord = Tuple[int, int]
Move = Tuple[int, Coord, Coord]
AODRound = List[Move]

def parse_layout(layout_dict: Dict[str, List[int]]) -> Dict[int, Coord]:
    return {int(k): (int(v[0]), int(v[1])) for k, v in layout_dict.items()}


def load_checkpoint(
    job_dir: Path,
) -> Tuple[Dict[int, Coord], List[dict], int, Dict[int, Coord]]:
    """Load compile state from ``job_dir/checkpoint.json``.

    Returns ``(initial_layout, schedule, completed_stages, current_layout)``,
    matching ``parse_job_logs`` so callers can use either path. Raises
    ``FileNotFoundError`` if no checkpoint exists; ``ValueError`` if it's
    present but malformed.
    """
    cp_path = job_dir / "checkpoint.json"
    if not cp_path.exists():
        raise FileNotFoundError(f"No checkpoint.json in {job_dir}")

    try:
        data = json.loads(cp_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse {cp_path}: {exc}") from exc

    required = ("initial_layout", "current_layout", "schedule", "completed_stages")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"{cp_path} missing fields: {missing}")

    try:
        initial_layout = parse_layout(data["initial_layout"])
        current_layout = parse_layout(data["current_layout"])
    except Exception as exc:
        raise ValueError(f"{cp_path} has malformed layout data: {exc}") from exc

    schedule = data["schedule"]
    if not isinstance(schedule, list):
        raise ValueError(f"{cp_path} schedule is not a list")

    try:
        completed = int(data["completed_stages"])
    except Exception as exc:
        raise ValueError(f"{cp_path} invalid completed_stages: {exc}") from exc

    # Each stage contributes an AOD entry + a cz_stage entry.
    if len(schedule) != 2 * completed:
        raise ValueError(
            f"{cp_path} inconsistent: schedule len {len(schedule)} "
            f"!= 2*completed_stages ({2 * completed})"
        )

    return initial_layout, schedule, completed, current_layout

def apply_aod_round(layout: Dict[int, Coord], round_moves: AODRound) -> Dict[int, Coord]:
    new_layout = dict(layout)
    for q, src, dst in round_moves:
        new_layout[q] = dst
    return new_layout

def extract_section(llm_log_text: str, section_header: str) -> str:
    """Return the text under ``section_header`` up to the next divider or EOF."""
    pattern = re.compile(r"={80}\n\s*" + re.escape(section_header) + r"\n={80}\n(.*?)((?=\n={80})|\Z)", re.DOTALL)
    m = pattern.search(llm_log_text)
    if not m:
        return ""
    return m.group(1).strip()

def parse_job_logs(job_dir: Path) -> Tuple[Dict[int, Coord], List[dict], int, Dict[int, Coord]]:
    output_path = job_dir / "output.txt"
    llm_log_path = job_dir / "llm_log.txt"
    
    if not output_path.exists() or not llm_log_path.exists():
        raise FileNotFoundError(f"Logs missing in {job_dir}")
        
    output_text = output_path.read_text(encoding="utf-8")
    llm_log_text = llm_log_path.read_text(encoding="utf-8")
    
    initial_layout = None
    if "[OK] Valid Initial layout generated" in output_text:
        matches = list(re.finditer(r"\[Initial Layout CZ Stage 1\] Attempt (\d+)/(\d+)", output_text))
        if matches:
            last_attempt = int(matches[-1].group(1))
            # CALL #N is monotone over the whole run, so match by attempt label only.
            pattern = re.compile(r"={80}\n\s*CALL #\d+ — Initial Layout Attempt " + str(last_attempt) + r" — OUTPUT\n={80}\n(.*?)((?=\n={80})|\Z)", re.DOTALL)
            m = pattern.search(llm_log_text)
            if m:
                content = m.group(1).strip()
                # find dict inside content
                dict_m = re.search(r"({[\s\S]*})", content)
                if dict_m:
                    res = ast.literal_eval(dict_m.group(1))
                    if "initial_layout" in res:
                        initial_layout = parse_layout(res["initial_layout"])
                        
    if initial_layout is None:
        raise ValueError("Could not find a valid initial layout in the logs.")
        
    current_layout = dict(initial_layout)
    schedule = []
    completed_stages_count = 0

    # Count completed stages from the per-stage OK messages.
    completed_matches = list(re.finditer(r"\[OK\] Transition to CZ Stage (\d+) completed", output_text))

    # Keep only the last success line per stage.
    unique_matches = {}
    for m in completed_matches:
         unique_matches[int(m.group(1))] = m
         
    for stage_num in sorted(unique_matches.keys()):
        m = unique_matches[stage_num]
        stage_idx = stage_num - 1 # 0-indexed
        
        target_attempt_match = list(re.finditer(rf"\[Target Layout CZ Stage {stage_idx+1}\] Attempt (\d+)/\d+\n.*?(?:\[OK\] Valid Target Layout generated!)", output_text, re.DOTALL))
        if not target_attempt_match:
            continue
            
        target_attempt = int(target_attempt_match[-1].group(1))
        
        target_pattern = re.compile(r"={80}\n\s*CALL #\d+ — Target Layout CZ" + str(stage_idx) + "→CZ" + str(stage_idx+1) + " Attempt " + str(target_attempt) + r" — OUTPUT\n={80}\n(.*?)((?=\n={80})|\Z)", re.DOTALL)
        llm_m = target_pattern.search(llm_log_text)
            
        if not llm_m:
            raise ValueError(f"Could not find Target Layout output for CZ{stage_idx}→CZ{stage_idx+1} attempt {target_attempt}")
            
        content = llm_m.group(1).strip()
        dict_m = re.search(r"({[\s\S]*})", content)
        if not dict_m:
             raise ValueError("Could not parse target layout dictionary")
        res = ast.literal_eval(dict_m.group(1))
        target_layout = parse_layout(res["target_layout"])
        formatted_target_layout = {str(q): [pos[0], pos[1]] for q, pos in sorted(target_layout.items())}
        
        opt_block = re.search(rf"\[Transition to CZ Stage {stage_idx+1}\].*?\[OPTIMIZE FINISHED\]", output_text, re.DOTALL)
        if not opt_block:
             print(f"No Optimize block found for CZ Stage {stage_idx+1}, assuming incomplete. Stopping recovery here.")
             break
             
        # Recover the optimizer's chosen round count for this stage.
        best_rounds_m = re.search(rf"\[Transition to CZ Stage {stage_idx+1}\].*?\[OPTIMIZE FINISHED\] Best schedule rounds: (\d+)", output_text, re.DOTALL)
        if not best_rounds_m:
             raise ValueError(f"Could not find 'Best schedule rounds' for CZ Stage {stage_idx+1}")
        best_rounds = int(best_rounds_m.group(1))

        if best_rounds == 0:
             schedule.append({
                 "target_layout": formatted_target_layout,
                 "aod_move_groups": []
             })
             schedule.append({"cz_stage": "PLACEHOLDER"})
             completed_stages_count += 1
             continue

        best_opt_attempt = None
        lines = opt_block.group(0).split('\n')
        current_attempt = None
        rounds_re = re.compile(rf'Rounds:\s*{best_rounds}(?:[^0-9]|$)')
        for line in lines:
            m = re.search(r'\[API Response\].*?\(Attempt (\d+)/\d+\)', line)
            if m:
                current_attempt = int(m.group(1))
            if '[ACCEPTED]' in line:
                best_opt_attempt = current_attempt
            if '-> Valid!' in line and rounds_re.search(line):
                best_opt_attempt = current_attempt
                break
                
        if best_opt_attempt is None:
             raise ValueError(f"No valid optimization found for CZ Stage {stage_idx+1} matching {best_rounds} rounds")
        
        opt_pattern = re.compile(r"={80}\n\s*CALL #\d+ — Optimize CZ" + str(stage_idx) + "→CZ" + str(stage_idx+1) + " Attempt " + str(best_opt_attempt) + r" — OUTPUT\n={80}\n(.*?)((?=\n={80})|\Z)", re.DOTALL)
        opt_m = opt_pattern.search(llm_log_text)
             
        if not opt_m:
             raise ValueError(f"Could not find Optimize output for CZ{stage_idx}→CZ{stage_idx+1} attempt {best_opt_attempt}")
             
        content = opt_m.group(1).strip()
        # Find json array
        json_m = re.search(r"(\[[\s\S]*\])", content)
        if not json_m:
             raise ValueError("Could not parse optimization array")
        optimized_raw = json.loads(json_m.group(1))
        
        formatted_aod_groups = []
        parsed_rounds = []
        for rnd in optimized_raw:
             moves = []
             formatted_rnd = []
             for m_val in rnd:
                  q = int(m_val[0])
                  src = (int(m_val[1][0]), int(m_val[1][1]))
                  dst = (int(m_val[2][0]), int(m_val[2][1]))
                  moves.append((q, src, dst))
                  formatted_rnd.append([q, [src[0], src[1]], [dst[0], dst[1]]])
             parsed_rounds.append(moves)
             formatted_aod_groups.append(formatted_rnd)
             
        schedule.append({
             "target_layout": formatted_target_layout,
             "aod_move_groups": formatted_aod_groups
        })
        schedule.append({"cz_stage": "PLACEHOLDER"})
        
        for rnd in parsed_rounds:
             current_layout = apply_aod_round(current_layout, rnd)
             
        completed_stages_count += 1
        
    return initial_layout, schedule, completed_stages_count, current_layout
