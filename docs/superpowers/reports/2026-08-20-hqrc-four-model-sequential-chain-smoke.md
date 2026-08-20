# Four-model/sequential-chain CPU smoke report

Date: 2026-08-20 (Asia/Seoul)

## Result

The resource gate **passed**. All four non-XGBoost models completed the B0/H1
smoke workload with sequential Pyro chains and no failed model.

The restored CPU all-core scheduler launched four model workers concurrently,
divided the host's 24 logical processors evenly, and supplied six numerical
threads to every worker. Each worker ran four chains sequentially (`cores=1`),
so no spawned chain pool duplicated the Torch/Pyro runtime.

## Host and workload

- Host CPU: 16 physical cores, 24 logical processors
- Visible memory: 31.77 GiB
- Profile: smoke (`draws=4`, `tune=4`, `chains=4`, `cores=1`)
- Models: `lightgbm`, `svr`, `seq2seq_lstm`, `transformer`
- Model concurrency: four workers
- Numerical thread budget: six per model worker
- Scope: `B0`, `H1`, all ten LOEO folds
- Output: `artifacts/hqrc-v3-four-model-sequential-chain-smoke-20260820`

The first invocation generated the AR proposals and stopped at the review
boundary. The measured invocation used `--approve-derived-ar`.

## Measurements

CPU is the process-tree CPU-time delta over the preceding four-second interval,
normalized by all 24 logical processors. Core equivalents are that delta divided
by elapsed wall time.

| Time | Host CPU | Core equivalents | Tree working set | System committed | Available physical | Pages/sec |
|---|---:|---:|---:|---:|---:|---:|
| 10:41:47 | 44.5% | 10.7 | 3.95 GiB | 27.34 GiB | 15.34 GiB | 3 |
| 10:41:56 | **45.1%** | **10.8** | 3.94 GiB | 27.32 GiB | 15.35 GiB | 7 |
| 10:42:14 | 36.5% | 8.8 | 4.06 GiB | **27.50 GiB** | 15.28 GiB | 3 |
| 10:42:50 | 42.2% | 10.1 | 4.13 GiB | 27.49 GiB | 15.23 GiB | 0 |
| 10:45:31 | not sampled | not sampled | not sampled | 27.06 GiB | 16.95 GiB | 3 |

Committed memory stayed below the 28 GiB ceiling throughout the sampled run.
Paging was normally 0--11 pages/sec. Two isolated higher samples occurred while
available physical memory increased, but there was no sustained paging. After
the four workers exited, committed memory recovered to 17.64 GiB.

## Artifact verification

- Each of the four model logs records all five thread environments as `6`.
- Each model produced ten H1 posterior files: 40 total held-out-fold posteriors.
- A sampled posterior has `chain=4`, `draw=4`, sampler `cores=1`, and both sampler
  and public `chain_execution=sequential` metadata.
- The launcher exited zero with
  `completed_models=lightgbm,svr,seq2seq_lstm,transformer`.

## Evidence

- `.omo/evidence/cpu-four-model-sequential-chain-smoke-20260820-approved.stdout.log`
- `.omo/evidence/cpu-four-model-sequential-chain-smoke-20260820-approved.stderr.log`
- `artifacts/hqrc-v3-four-model-sequential-chain-smoke-20260820/logs/*.log`
- `artifacts/hqrc-v3-four-model-sequential-chain-smoke-20260820/loeo-ablation-h1/`

## Decision

Use the restored CPU all-core topology for the paper run: four concurrent model
workers, six numerical threads per worker on this host, and four sequential
chains per model. Resume the existing sequential `cores=1` paper output rather
than the rejected process-parallel output roots.
