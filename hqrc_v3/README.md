# HQRC v3 experiment workflow

HQRC v3 keeps baseline forecasting and residual correction evaluation separate.  The
four expanding OOF fits (`2019→2020`, …, `2019–2022→2023`) are internal baseline
fits that create honest 2020–23 residual targets for HQRC.  The final baseline is a
separate single `2019–2023→2024` refit and is the sole causal 2024 forecast.  OOF
cache reuse reduces computation; it is not a second final evaluation.

Run the stages in order:

```text
hqrc audit-data --data power_demand_final.csv --fixed-bounds
hqrc tune-baselines --data power_demand_final.csv --config configs/experiment.toml
hqrc generate-oof --data power_demand_final.csv --config configs/experiment.toml ...
hqrc diagnose-ar --residuals runs/<id>/residuals/events.parquet ... --through 2023
hqrc approve-ar-calibration runs/<id>/ar_diagnostics/proposed.json --output ...
hqrc fit-final-baselines --data power_demand_final.csv --config configs/experiment.toml ...
hqrc fit-corrections --run-dir runs/<id> --config configs/experiment.toml --approved-ar ... --evaluation causal-2024 --seed 11 --profile paper
hqrc run-ablations --run-dir runs/<id> --config configs/experiment.toml --approved-ar ... --seed 11 --profile paper
hqrc benchmark-samplers --run-dir runs/<id> --config configs/experiment.toml --approved-ar ... --seed 11 --draws 1000 --tune 1000 --chains 4 --profile paper
hqrc report --run-dir runs/<id> --profile paper
```

`smoke` output is explicitly non-paper.  Only a `paper` profile that passes all
posterior diagnostics may populate manuscript numbers.  Every run contains
`manifest.json`, prediction/residual/AR/posterior artifacts, normalized `metrics/`
tables, `figures/`, sampler benchmark JSON, and an atomically written `COMPLETE`.
Event and probabilistic metric tables map to the main results tables; pooling and
ablation tables map to sensitivity tables; sampler results map to the computational
appendix.  Values are never invented by reporting.

Nutpie is optional.  It can become the default only when the recorded benchmark
agreement/diagnostic and ≥20% speed or bulk-ESS/s gates pass.  Polars/Arrow
zero-copy and parallel metadata are benchmarked too.  No custom Rust/PyO3 rewrite is
permitted unless measured preprocessing/postprocessing is at least 20% of total wall
time and copying is confirmed as the bottleneck.

Use `pytest -m "not slow"` for fast contracts and the explicit `tests/slow` command
for the real-data audit/classical-baseline smoke; neither command is a full paper run.
