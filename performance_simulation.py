"""Neutral-atom hardware simulator.

Computes hardware runtime ``T_exe`` and end-to-end fidelity ``F`` for
a sequence of AOD move groups, with fidelity broken into the four
components fed back to the optimizer (transfer, coherence, CZ gate,
idle-during-gate). Constants and the accounting model match the
PowerMove and Enola simulators so metrics are directly comparable.
"""

# Constants are aligned with the PowerMove / Enola simulators.
N_Block = 1
X_SEP = 19          # μm between adjacent groups along x
Y_SEP = 15          # μm between rows
SITE_SEP = 1.5      # μm between left and right trap within a group
Fidelity_2Q_Gate = 0.995
Fidelity_Atom_Transfer = 0.999
Coherence_Time = 1.5e6  # μs
MUS_PER_FRM = 8         # μs per AOD frame (transfer overhead unit)

def calculate_transition_performance(aod_move_groups, n_qubits, cz_stages=None):
    """Compute T_exe and the fidelity decomposition for an AOD schedule.

    ``aod_move_groups`` is a list of rounds, each a list of
    ``(q, (src_x, src_y), (dst_x, dst_y))`` moves. ``cz_stages`` is
    optional; when supplied, CZ gate and idle-during-gate fidelities
    are included in the result.
    """
    cir_fidelity_atom_transfer = 1.0
    cir_fidelity_coherence = 1.0
    cir_qubit_idle_time = [0.0] * n_qubits

    total_transfer_duration = 0.0
    total_movement_duration = 0.0

    # Active set per round, used for chain-detection and idle accounting.
    active_sets = []
    for round_moves in aod_move_groups:
        s = set()
        for m in round_moves:
            s.add(m[0])
        active_sets.append(s)

    for i, round_moves in enumerate(aod_move_groups):
        current_active = active_sets[i]
        active_qubits = current_active
        max_distance = 0.0

        # Round duration is set by the longest single move (parallel AOD).
        # Split-trap mapping: cell_x = x // 2, slot = x % 2, so
        # physical Δx = |Δcell| * X_SEP + |Δslot| * SITE_SEP.
        for move in round_moves:
            q_idx, src, dst = move
            src_cell_x, src_slot = src[0] // 2, src[0] % 2
            dst_cell_x, dst_slot = dst[0] // 2, dst[0] % 2

            dx_physical = abs((dst_cell_x - src_cell_x) * X_SEP + (dst_slot - src_slot) * SITE_SEP)
            dy_physical = abs((dst[1] - src[1]) * Y_SEP)
            distance = dx_physical + dy_physical

            if distance > max_distance:
                max_distance = distance

        # Transfer fidelity is paid per pickup. An atom moving in
        # consecutive rounds is treated as a single chain and pays once.
        num_new_pickups = 0
        if i == 0:
            num_new_pickups = len(active_qubits)
        else:
            prev_active = active_sets[i-1]
            for q in active_qubits:
                if q not in prev_active:
                    num_new_pickups += 1

        cir_fidelity_atom_transfer *= pow(Fidelity_Atom_Transfer, num_new_pickups)

        if max_distance > 0:
            move_duration = 200 * (max_distance / 110)**0.5
        else:
            move_duration = 0.0
        total_movement_duration += move_duration

        # Skip a pickup if this round's movers are a subset of the
        # previous round's; skip a dropoff if of the next round's.
        transfer_time = 2 * MUS_PER_FRM
        if i > 0:
            if current_active.issubset(active_sets[i-1]):
                transfer_time -= MUS_PER_FRM
        if i < len(aod_move_groups) - 1:
            if current_active.issubset(active_sets[i+1]):
                transfer_time -= MUS_PER_FRM

        total_transfer_duration += transfer_time

        # Coherence: accumulate idle time only for NON-moving atoms.
        round_duration = move_duration + transfer_time
        for q in range(n_qubits):
            if q not in active_qubits:
                cir_qubit_idle_time[q] += round_duration

    # Clamp per-qubit base to [0, 1]: idle time can exceed Coherence_Time
    # on very long schedules, which would otherwise produce a negative
    # (unphysical) fidelity contribution.
    for t in cir_qubit_idle_time:
        base = max(0.0, 1.0 - t / Coherence_Time)
        cir_fidelity_coherence *= base ** (N_Block ** 2)

    cir_fidelity_2q_gate = 1.0
    cir_fidelity_2q_gate_for_idle = 1.0
    if cz_stages is not None:
        # Idle qubits during a CZ pulse pay roughly half the 2Q gate error
        # (they see the global beam but not the full interaction).
        fidelity_2q_gate_for_idle = 1 - (1 - Fidelity_2Q_Gate) / 2  # 0.9975
        for stage in cz_stages:
            n_pairs = len(stage)
            cir_fidelity_2q_gate *= pow(Fidelity_2Q_Gate, n_pairs)
            idle_qubits = n_qubits - 2 * n_pairs
            cir_fidelity_2q_gate_for_idle *= pow(fidelity_2q_gate_for_idle, idle_qubits * (N_Block ** 2))

    final_fidelity = (cir_fidelity_2q_gate * cir_fidelity_2q_gate_for_idle 
                      * cir_fidelity_atom_transfer * cir_fidelity_coherence)
    total_duration = total_movement_duration + total_transfer_duration

    return {
        "total_duration": total_duration,
        "final_fidelity": final_fidelity,
        "movement_fidelity": cir_fidelity_atom_transfer,
        "coherence_fidelity": cir_fidelity_coherence,
        "gate_fidelity": cir_fidelity_2q_gate,
        "gate_idle_fidelity": cir_fidelity_2q_gate_for_idle,
        "transfer_duration": total_transfer_duration,
        "movement_duration": total_movement_duration
    }

if __name__ == '__main__':
    import sys
    import json
    import os
    
    if len(sys.argv) < 2:
        print("Usage: python performance_simulation.py <compiled_results.json>")
        sys.exit(1)
        
    import re

    input_file = sys.argv[1]
    with open(input_file, 'r') as f:
        data = json.load(f)

    results_dict = data.get('results', {})

    # Infer n_qubits from group name (e.g. "q20_regular5" -> 20).
    group_name = data.get('group', '')
    m = re.match(r'q(\d+)', group_name)
    default_n_qubits = int(m.group(1)) if m else None

    all_metrics = {}

    for circuit_id, item in results_dict.items():
        if isinstance(item, dict):
            schedule = item.get('schedule', [])
            n_qubits = item.get('n_qubits', default_n_qubits)
            if n_qubits is None:
                print(f"[ERROR] Cannot determine n_qubits for {circuit_id}, skipping.")
                continue

            all_move_groups = []
            cz_stages = []
            for stage in schedule:
                if 'aod_move_groups' in stage:
                    all_move_groups.extend(stage['aod_move_groups'])
                if 'cz_stage' in stage:
                    cz_stages.append(stage['cz_stage'])

            metrics = calculate_transition_performance(all_move_groups, n_qubits, cz_stages=cz_stages)
            all_metrics[circuit_id] = metrics
            
            print(f"Results for Circuit: {circuit_id}")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.6f}")
                else:
                    print(f"  {k}: {v}")
            print("-" * 40)
            
    base_name, ext = os.path.splitext(input_file)
    output_file = f"{base_name}_performance{ext}"
    with open(output_file, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Performance metrics saved to: {output_file}")

