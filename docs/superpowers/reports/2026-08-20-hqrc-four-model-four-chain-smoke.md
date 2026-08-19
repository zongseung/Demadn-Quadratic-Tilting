# Four-model/four-chain CPU smoke report

Date: 2026-08-20 (Asia/Seoul)

## Result

The resource gate **failed**. The detached paper run was not started.

The Windows-native smoke launched all four non-XGBoost models with the intended
CPU environment (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` all set to `1`). Each Pyro
model requested four spawned chain workers.

## Host and command

- Host CPU: 16 physical cores, 24 logical processors
- Visible memory: 31.77 GiB
- Pre-smoke system committed memory: 18.88 GiB
- Profile: smoke (`draws=4`, `tune=4`, `chains=4`, CPU-resolved `cores=4`)
- Models: `lightgbm`, `svr`, `seq2seq_lstm`, `transformer`
- Scope: `B0`, `H1`
- Output: `artifacts/hqrc-v3-four-chain-smoke-20260820`

## Observations

| Time | Observed state | Tree working set | System committed | Free physical | Pages/sec |
|---|---|---:|---:|---:|---:|
| 03:11:37 | Four model workers active; no chain child caught in sample | 3.96 GiB | 27.06 GiB | not sampled | 23 |
| 03:11:53 | Four model workers active | 3.98 GiB | 27.44 GiB | 14.46 GiB | 805 |
| 03:12:05 | Four model workers active | 4.05 GiB | 27.36 GiB | 14.46 GiB | 3 |
| 03:12:28 | Four spawned Pyro chain children observed under one model | 6.27 GiB | **31.20 GiB** | 12.80 GiB | 0 |

The approved ceiling was 28 GiB committed memory. One observed four-chain pool
already exceeded it by 3.20 GiB while the other model workers remained resident.
The smoke draws are very short, so the other models' chain pools completed between
samples; their ArviZ diagnostic output is present in the durable model logs. A
paper run would keep the chain pools alive much longer and can overlap them, so
continuing toward sixteen simultaneous chain children was unsafe.

Three posteriors completed before termination (`lightgbm`, `seq2seq_lstm`, and
`transformer`, first held-out fold). Each has dimensions `chain=4`, `draw=4`,
sampler metadata `chains=4`, `cores=4`, `chain_execution=parallel`, and the public
attribute `hqrc_chain_execution=parallel`. This confirms that the implemented
parallel path produced valid, correctly identified artifacts before the resource
gate stopped the broader run.

The exact smoke process tree was terminated after the gate failure. No matching
process remained. Immediately afterward, committed memory recovered to 19.48 GiB
and free physical memory to 18.02 GiB. The `svr` `BrokenProcessPool` entry is the
expected consequence of that intentional termination, not an independently
observed sampler failure.

## Evidence

- `.omo/evidence/cpu-four-chain-smoke-20260820-retry1.stdout.log`
- `.omo/evidence/cpu-four-chain-smoke-20260820-retry1.stderr.log`
- `artifacts/hqrc-v3-four-chain-smoke-20260820/logs/*.log`

The initial detached wrapper attempt is retained separately as
`.omo/evidence/cpu-four-chain-smoke-20260820.*.log`; it failed before model launch
because the monitoring wrapper passed the PowerShell model array as one comma-
joined value. The successful retry used PowerShell-native array parsing.

## Decision

Do not launch `artifacts/hqrc-v3-loeo-cpu-four-chain-20260820` with four models by
four chains on this 32 GiB host. The code path is implemented and testable, but
this particular concurrency topology does not satisfy the host memory gate.

## Regression verification

The complete non-slow suite finished after the resource-gate smoke. After the
final code review fixes (device/topology enforcement and fail-fast pool
termination), the full suite was rerun on the integration candidate:

```text
840 passed, 34 skipped, 11 deselected, 78 warnings in 2889.50s (0:48:09)
```

Durable logs:

- `.omo/evidence/cpu-four-chain-full-regression-20260820.stdout.log`
- `.omo/evidence/cpu-four-chain-full-regression-20260820.stderr.log`
- `.omo/evidence/cpu-four-chain-review-fixes-full-regression-20260820.stdout.log`
- `.omo/evidence/cpu-four-chain-review-fixes-full-regression-20260820.stderr.log`
