# Qubrio

Official code for *Qubrio: High-Performance Quantum Compilation via Multi-Agent
LLM Collaboration* (NeurIPS 2026).

Jixuan Ruan\*, Zhuo Cui\*, Zhengding Hu, Zhongkai Yu, Xiang Fang, Yue Guan,
Jason Ludmir, Phillip E. Weinberg, Xiu-Zhe Luo, Sheng-Tao Wang,
Sonia Lopez Alarcon, Hezi Zhang, Yufei Ding (\*equal contribution)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in the API key for the backbone you use
```

## Benchmarks

`benchmark/` ships the eight QASMBench-derived circuits used: QFT-18, QFT-63, QAOA-20, QAOA-60, GHZ-23, GHZ-78, Ising-26,
Ising-66.

## Running

End to end — compile → aggregate → simulate:

```bash
python run_all.py --test-dir benchmark --provider deepseek --workers 4
python pipeline.py --skip-compile --log-root logs/<timestamp>
```

`run_all.py` writes one `result.json` (and `result_unoptimized.json`)
per circuit to `logs/<timestamp>/<group>/<circuit>/`. `pipeline.py`
aggregates these and reports hardware runtime `T_exe` and end-to-end
circuit fidelity `F` using the simulator from PowerMove + Enola.

For a quick sanity check, truncate each circuit to its first N stages:

```bash
python run_all.py --test-dir benchmark --provider qwen --max-stages 2
```

If a long run is interrupted, `resume_all.py` picks it up from the
per-stage checkpoints written alongside the logs.

Defaults: temperature 0.7, top-p 1.0, 10 retry rounds
for placement and routing, 5 rounds for the optimizer.

## Layout

- `incremental_compiler.py` — multi-agent compilation loop (entry point for the algorithm)
- `verify.py` — deterministic placement and routing verifiers
- `performance_simulation.py` — neutral-atom hardware simulator (`T_exe`, fidelity)
- `prompt_*.py` — prompt templates for each agent
- `api_client.py` — unified LLM client
- `run_all.py`, `resume_all.py`, `pipeline.py` — CLI runners
- `aggregate_results.py`, `log_parser.py` — output post-processing

## Citation

```bibtex
@inproceedings{ruan2026qubrio,
  title     = {Qubrio: High-Performance Quantum Compilation via Multi-Agent LLM Collaboration},
  author    = {Ruan, Jixuan and Cui, Zhuo and Hu, Zhengding and Yu, Zhongkai and Fang, Xiang and
               Guan, Yue and Ludmir, Jason and Weinberg, Phillip E. and Luo, Xiu-Zhe and
               Wang, Sheng-Tao and Lopez Alarcon, Sonia and Zhang, Hezi and Ding, Yufei},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
