# Task 14 implementation report

Status: READY FOR INDEPENDENT REVIEW

Implementation commits:

- `da1aa93` (`feat(hqrc-v3): bind causal correction inputs`)
- `0d4fdfa` (`feat(hqrc-v3): fit causal 2024 corrections`)

## Delivered

- Made the digest-validated approved AR artifact expose and revalidate its opaque
  `EventResidualContext`, and preserved that complete approval binding in NetCDF metadata and
  across the sampler-process worker boundary.
- Added a strict causal adapter for the exact eight 2020--2023 expanding-OOF holiday
  occurrences. It rejects incomplete, duplicate, extra, final-2024, non-hourly, reordered, or
  context-mismatched inputs and uses only the finite positive `oof-2023` non-event scale.
- Rebuilds the audited source feature matrices and invokes the public final-baseline stage only
  in validation/reuse mode. The production stage requires `fit_count == 0`, checks all six
  source identities, and verifies the full 7,320-hour final stream against source-derived truth.
- Fits only the frozen primary H3 specification: partial pooling, full between-event covariance,
  restriction covariate, and Gaussian event-reset AR(1), with the transformed Beta prior loaded
  from the approved context. No 2024 outcome enters fitting or calibration.
- Predicts exactly the registered 2024 Seollal (144 hours) and Chuseok (120 hours) windows.
  Predictive draws are exactly `baseline + sigma_N * (q + e)` with independent AR resets, while
  the point stream is bitwise identical to the final baseline outside those 264 hours. The
  2024-10-01 temporary holiday is not corrected.
- Added immutable context-specific publication, strict diagnostic gates, fsynced atomic products,
  complete-result reuse, and hash-valid posterior-checkpoint resume without resampling. Partial,
  symlinked, diagnostically invalid, or semantically changed publications fail closed.
- Separated provenance for the source publication profile and sampler profile. A paper sampler
  requires paper sources; an explicitly reduced smoke sampler may validate either paper or smoke
  sources, and paper artifacts are never relabeled.
- Wired only `fit-corrections --evaluation causal-2024`; `loeo` explicitly refuses to reuse the
  eight-event approval because fold-specific approvals are a later task. CLI `--seed` controls
  sampler/predictive randomness, never baseline stream selection.

## TDD and verification evidence

- Genuine RED covered the previously absent production correction stage and unsafe approved-AR
  context substitution before the implementation was added.
- Focused approval/context, causal stage, and CLI suite:
  `uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml
  hqrc_v3/tests/integration/test_ar_artifact.py
  hqrc_v3/tests/unit/test_correction_stage.py
  hqrc_v3/tests/integration/test_cli_corrections.py -q`
  — `22 passed, 3 expected reduced-draw warnings in 1.93s`.
- Adjacent report-pipeline regression: `15 passed`.
- Full non-slow suite:
  `uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests
  -m 'not slow' -q`
  — `411 passed, 8 deselected, 50 warnings in 62.94s`.
- Full project Ruff:
  `uv run --project hqrc_v3 --locked ruff check --config hqrc_v3/pyproject.toml
  hqrc_v3/src hqrc_v3/tests`
  — clean.
- `git diff --check` — clean.
- The protected untracked `hqrc_v3/uv.lock` remains byte-identical at
  `f07f2944707750a9b0753690e6fca2d9c83dbf1d29340e3628483573a2766657`.

## Reduced real-source proof

- Ran the opt-in slow test against the completed real paper publication and the approved
  LightGBM-B1 AR artifact. It rebuilt/revalidated source truth, proved final-baseline reuse with
  no fit, traversed the actual H3 PyMC model with explicit reduced limits, bound eight training
  and two evaluation occurrences, and verified event-only correction.
- Result: `1 passed, 24 expected reduced-draw ArviZ warnings in 12.17s`.
- The test compared the protected baseline manifest, final members/point forecasts, residual
  manifest/data, and approved AR artifact hashes before and after. All remained unchanged; the
  correction output was written only beneath pytest's temporary directory.

## Review boundary and concerns

- Full four-chain paper sampling remains prohibited until a fresh independent review returns
  READY. This implementation does not claim or publish paper correction results.
- LOEO, H0--H5 ablations, pooling ablations, alternative innovation models, and paper-wide
  reporting remain intentionally outside this bounded task.
- The 50 fast-suite and 24 real-smoke warnings arise from intentionally tiny synthetic/reduced
  posterior draws and are not present as accepted paper diagnostic evidence; paper publication
  still requires the frozen strict R-hat, ESS, and divergence gates.
