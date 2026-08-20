# Two-model/four-chain/two-thread CPU smoke report

Date: 2026-08-20 (Asia/Seoul)

## Result

The experiment **failed both objectives**: it did not saturate the CPU and it
exceeded the memory gate. The experimental production and test changes were
removed; the checked-in scheduler remains at two concurrent models, four chains
per active model, and one thread per chain.

## Hypothesis

Allowing two Torch intra-op threads in each of the eight active chain processes
would expose up to 16 runnable compute threads on the 16-physical-core,
24-logical-processor target. The scheduler temporarily propagated `2` through
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS`; each Pyro child temporarily
used that value for `torch.set_num_threads` while retaining one inter-op thread.

The behavior was developed test-first. The focused RED run failed in all three
expected places under the one-thread implementation. The experimental GREEN run
then passed, followed by the scheduler and sampler files completing with
`109 passed` and Ruff format/check success.

## Host and workload

- Host CPU: 16 physical cores, 24 logical processors
- Visible memory: 31.77 GiB
- Profile: smoke (`draws=4`, `tune=4`, `chains=4`, `cores=4`)
- Active queues: `lightgbm -> seq2seq_lstm` and `svr -> transformer`
- Experimental topology: `2 models x 4 chains x 2 intra-op threads`
- Scope: `B0`, `H1`
- Output: `artifacts/hqrc-v3-two-model-four-chain-two-thread-smoke-20260820`

## Measurements

CPU is the process-tree CPU-time delta over the preceding four-second interval,
normalized by all 24 logical processors. Core equivalents are the same delta
divided by elapsed wall time.

| Time | Processes | Host CPU | Core equivalents | Tree working set | System committed | Available physical | Pages/sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10:19:27 | 12 | 11.4% | 2.7 | 4.72 GiB | 27.57 GiB | 14.93 GiB | 3 |
| 10:19:32 | 8 | 17.8% | 4.3 | 2.19 GiB | 23.08 GiB | 16.88 GiB | 3 |
| 10:19:36 | 8 | 21.9% | 5.2 | 2.22 GiB | 23.11 GiB | 16.86 GiB | 0 |
| 10:19:40 | 12 | 26.0% | 6.2 | 2.53 GiB | 23.76 GiB | 16.55 GiB | 0 |
| 10:19:45 | 16 | **28.0%** | **6.7** | 7.07 GiB | **31.68 GiB** | 13.22 GiB | 3 |

The model logs confirm all five numerical thread environment values were `2` for
both active models. Nevertheless, two threads are only an upper bound: the small
HQRC tensor operations did not expose enough intra-op work for Torch to use the
second thread consistently. Compared with the one-thread two-model smoke, peak
committed memory worsened from 31.27 GiB to 31.68 GiB while observed CPU remained
well below saturation.

The run was stopped when committed memory exceeded the unchanged 28 GiB ceiling.
All matching processes and descendants were terminated. No process matching the
unique output root remained; committed memory recovered to 17.94 GiB and
available physical memory to 18.89 GiB. `lightgbm` and `svr` each completed two
H1 posterior files before termination; second-queue models did not start.

## Evidence

- `.omo/evidence/cpu-two-model-four-chain-two-thread-smoke-20260820-approved.stdout.log`
- `.omo/evidence/cpu-two-model-four-chain-two-thread-smoke-20260820-approved.stderr.log`
- `artifacts/hqrc-v3-two-model-four-chain-two-thread-smoke-20260820/logs/*.log`
- `artifacts/hqrc-v3-two-model-four-chain-two-thread-smoke-20260820/loeo-ablation-h1/`

## Decision

Reject the two-thread-per-chain change. Full CPU utilization cannot be obtained
from this workload merely by increasing Torch's intra-op limit. On this host the
binding constraint is the memory cost of independent spawned chain processes;
adding more models, chains, or fold processes would exceed the existing resource
gate unless per-process memory is reduced first.
