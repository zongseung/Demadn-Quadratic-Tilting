# HQRC v3 operator boundary

The CLI deliberately exposes only stages with a complete artifact contract.  These
are runnable now:

```text
hqrc audit-data --data power_demand_final.csv --fixed-bounds
hqrc diagnose-ar --residuals runs/RUN_ID/inputs/standardized_residuals.parquet --output runs/RUN_ID/ar_diagnostics/proposed.json --residual-sha256 RESIDUAL_SHA256 --config-sha256 CONFIG_SHA256 --event-sha256 EVENT_SHA256 --through 2023
hqrc approve-ar-calibration --proposal runs/RUN_ID/ar_diagnostics/proposed.json --output runs/RUN_ID/ar_diagnostics/approved.json --residual-sha256 RESIDUAL_SHA256 --config-sha256 CONFIG_SHA256 --event-sha256 EVENT_SHA256
hqrc report --run-dir runs/RUN_ID --profile smoke
```

`RUN_ID` is a directory prepared by the Python adapter boundary.  Before `report`,
it must contain the strict version-2 manifest and its run-local source audit,
resolved config, event registry, standardized residuals, validated OOF predictions,
approved AR artifact, NetCDF InferenceData, event metrics, and sampler benchmark.
Every manifest digest is recalculated from those files.  Reporting reloads the AR
approval using the actual residual/config/event hashes, removes any stale `COMPLETE`,
stages reports, then writes a JSON completion marker bound to the final manifest.

The four OOF fits (`2019→2020` through `2019–2022→2023`) create internal residual
targets.  The separate `2019–2023→2024` baseline refit is the only causal 2024
baseline.  `fit-corrections`, `run-ablations`, and `benchmark-samplers` are Python
adapter boundaries until a serialized `HQRCData`/frozen-model loader is supplied;
they do not claim to execute from partial CLI inputs.  A smoke report is labeled
non-paper.  Paper reporting is rejected until the full normalized manuscript-table
suite and actual four-chain, 1000-draw/tune posterior diagnostics are present.

Run fast contracts with `pytest -m "not slow"`.  The slow smoke audits the real
51,144-row source and fits/caches a small real SVR OOF prediction; it is not a paper
experiment.
