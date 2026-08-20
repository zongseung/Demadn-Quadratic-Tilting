# Two-model/four-chain CPU smoke report

Date: 2026-08-20 (Asia/Seoul)

## Result

The resource gate **failed**. The paper run was not started.

The CPU scheduler was capped at two concurrent model queues while retaining all
four non-XGBoost models. Each active model requested four spawned,
single-threaded Pyro chains, for a maximum sampling topology of
`2 models x 4 chains x 1 thread`.

## Host and workload

- Host CPU: 16 physical cores, 24 logical processors
- Visible memory: 31.77 GiB
- Profile: smoke (`draws=4`, `tune=4`, `chains=4`, CPU-resolved `cores=4`)
- Models: `lightgbm`, `svr`, `seq2seq_lstm`, `transformer`
- Concurrent queues: `lightgbm -> seq2seq_lstm` and `svr -> transformer`
- Scope: `B0`, `H1`
- Output: `artifacts/hqrc-v3-two-model-four-chain-smoke-20260820`

The first invocation created the required AR proposals and exited at the review
boundary. The second invocation used `--approve-derived-ar` and performed the
sampling measured below.

## Observations

| Time | Process-tree state | Tree working set | System committed | Available physical | Pages/sec |
|---|---|---:|---:|---:|---:|
| 09:59:44 | Two model workers resident; chain pools not yet observed | 2.15 GiB | 22.74 GiB | 16.94 GiB | 7 |
| 09:59:58 | Eight additional chain processes entering the tree | 4.89 GiB | 27.30 GiB | 14.78 GiB | 0 |
| 10:00:01 | Both four-chain pools overlapped | 7.30 GiB | **31.27 GiB** | 13.07 GiB | 0 |
| 10:00:05 | One chain pool had completed | 4.74 GiB | 27.13 GiB | 14.95 GiB | 3 |

The approved ceiling was 28 GiB committed memory. The overlapping two-model
chain pools exceeded it by 3.27 GiB. The exact run was stopped, including
orphaned descendants left after the detached wrapper exited. No process matching
the unique output root remained. After cleanup, committed memory was 18.00 GiB
and available physical memory was 18.82 GiB.

Before termination, `lightgbm` and `svr` each completed three H1 held-out-fold
posteriors. Their summaries each contain 16 samples, consistent with four chains
and four retained smoke draws. `seq2seq_lstm` and `transformer` did not begin
sampling because they were second in their respective queues.

## Evidence

- `.omo/evidence/cpu-two-model-four-chain-smoke-20260820.stdout.log`
- `.omo/evidence/cpu-two-model-four-chain-smoke-20260820.stderr.log`
- `.omo/evidence/cpu-two-model-four-chain-smoke-20260820-approved.stdout.log`
- `.omo/evidence/cpu-two-model-four-chain-smoke-20260820-approved.stderr.log`
- `artifacts/hqrc-v3-two-model-four-chain-smoke-20260820/logs/*.log`
- `artifacts/hqrc-v3-two-model-four-chain-smoke-20260820/loeo-ablation-h1/`

## Decision

Do not launch the two-model/four-chain CPU paper run on this 32 GiB host. The
scheduler cap is implemented and verified, but this topology does not satisfy
the existing memory gate. The next lower model-level topology is one active
model with four chains.
