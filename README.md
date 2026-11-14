# ShadowVeil

*ShadowVeil: Prospective Counterfactual Revocation for Mamba Inference State.*
Exact counterfactual revocation of Mamba recurrent state under a known expiry, with a lookahead-scheduled shadow
(`delta` interpolates between replay at the deadline and a sanitized shadow kept through the whole window).

## Authors
Egypt Japan University of Science and Technology (EJUST), Egypt

| Author | Email |
|---|---|
| Abdelrahman A. Alshames (corresponding) | abdelrahman.aboegela@ejust.edu.eg |
| Aly H. Gouda | aly.gouda@ejust.edu.eg |
| Omnia S. Eshra | omnia.eshra@ejust.edu.eg |
| Zeina Eltannikhy | zeina.eltannikhy@ejust.edu.eg |
| Salma A. Alashry | salma.alashry@ejust.edu.eg |

## Layout
```
paper/          LaTeX source (IEEEtran), TikZ overview, generated numbers.tex, figures/
src/shadowveil/ runtime library: cache operations (chunked.py), fused Triton continuation (fused.py), online runtime (runtime.py),
                host-load records (hostload.py), atomic result writing (iohelp.py)
experiments/
  mechanism/    exactness: non-zero snapshots, batch geometry, fused lookahead sweep, batched shadow rows, multi-item scaling,
                online runtime, reconvergence, deadline chains, memory scaling, Mamba-2 portability
  cost/         retirement latency, cost profiles, deadline latency
  scheduling/   scheduler simulation, real serving harness, simulation-vs-harness validation
  audit/        protocol runs (TAB, Enron), attackers and controls, equivalence statistics, distance to the counterfactual
  reporting/    aggregation, verification, tables, figures, paper numbers
tools/          pipeline.py (ordered, resumable runner), review_status.py
results/        result JSON files; prespecified confirmation runs are confirm_*.json
artifacts/      aggregate summary, CSV and verification manifest
docs/           prespecified protocol, literature note
data/           TAB and CMU Enron subsets (not tracked; see licences)
```

## Setup
```powershell
. .\env.ps1                 # PYTHONPATH, offline model cache, Triton compiler path
pip install -r requirements.txt
```
Models (Hugging Face cache, offline): `state-spaces/mamba-{130m,370m,790m,1.4b}-hf`, `AntonV/mamba2-{130m,370m,780m}-hf`; tokenizer
`EleutherAI/gpt-neox-20b`. Reference hardware: RTX 3060 Laptop (6 GB) for all timings, protocol seeds and exactness checks; an RTX 3050 (8 GB)
ran the 1.4B and Mamba-2 attacker audits, the nonlinear probes and the counterfactual-distance runs (no timings). The fused path needs Triton
(`triton-windows` on Windows, `triton` on Linux). The selective-scan comparison uses the official `mamba_ssm` kernel in a WSL environment.

## Run
```powershell
python tools/pipeline.py status            # which steps are done
python tools/pipeline.py run --smoke       # one tiny run of every script, writes only to scratch/smoke
python tools/pipeline.py run               # everything; resumable after any interruption; timings wait for a quiet host
python tools/run_all.py                    # restarts the pipeline until every step is done (unattended runs)
python experiments/reporting/verify_shadowveil_artifacts.py
cd paper; pdflatex shadowveil-paper.tex; pdflatex shadowveil-paper.tex
```
Every timing result records the host CPU load; a result taken on a loaded host is rerun, never cited.

## Data licences
TAB: MIT. CMU Enron annotated subsets: no standard open licence; review their terms before redistribution (`data/` is not tracked).
