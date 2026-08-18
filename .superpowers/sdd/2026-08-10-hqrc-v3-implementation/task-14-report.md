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

## Independent-review fix round 1 — RED evidence

- Before production edits, reviewer reproductions were added for all five downstream crash
  boundaries, independently self-rehashed tampering of all four Parquet products, sampler
  profile/RNG namespace collision, and a Bayesian `SamplingError` escaping the CLI boundary.
- Exact RED command:
  `uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml
  hqrc_v3/tests/unit/test_correction_stage.py
  hqrc_v3/tests/integration/test_cli_corrections.py -q`.
- Result: `11 failed, 13 passed, 16 warnings in 2.27s`. The failures independently demonstrated
  five non-resumable product/manifest crashes, acceptance of four semantically changed and fully
  rehashed Parquets, one smoke/RNG namespace collision, and one uncaught diagnostic error.

### Fix round 1 GREEN

Commit: `5031417` (`fix(hqrc-v3): harden correction publication reuse`)

- A valid HQRCData/posterior checkpoint now resumes after each known product or manifest boundary,
  removes only explicitly enumerated downstream files, regenerates them deterministically, and
  never calls the sampler again. Unknown entries, symlinks, and invalid HQRCData/posteriors remain
  fail-closed.
- Complete reuse reloads the diagnostics-valid posterior and re-derives all four products from the
  manifest-bound sampler seed and draw count. Exact column order, schema, row order, scalar/list
  values, masks, forecasts, and metrics must match, so fully rehashed semantic tampering fails.
- Output paths now include both sampler profile and sampler RNG seed below the approval-derived
  baseline seed. Smoke, paper, and distinct sampler seeds are independent namespaces.
- `SamplingError` now uses the declared `hqrc:` stderr and exit-code-2 CLI boundary without
  broadening it to unrelated runtime/programmer exceptions.
- Focused approval/correction/CLI suite: `37 passed, 25 expected tiny-draw warnings in 3.36s`.
- Full non-slow suite: `426 passed, 8 deselected, 72 expected tiny-draw warnings in 63.40s`.
- Full project Ruff and `git diff --check`: clean. Protected nested lock SHA remains
  `f07f2944707750a9b0753690e6fca2d9c83dbf1d29340e3628483573a2766657`.
- The real paper-source/reduced-PyMC smoke now also performs a second complete semantic reuse from
  NetCDF without sampling: `1 passed, 36 expected reduced-draw warnings in 13.21s`. Protected
  paper inputs remained hash-identical and all correction output stayed in pytest temporary space.
- Full paper sampling and actual paper correction publication remain prohibited pending a fresh
  independent READY verdict.

## Independent-review fix round 2 — RED evidence

- Added direct reviewer reproductions for a self-rehashed logical-output alias and for partial
  resume with current-generation NPZ/metadata symlinks plus unknown regular/symlink namespace
  entries. The tests also require zero new sampler calls and no invalid `COMPLETE` publication.
- Exact RED command:
  `uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml
  hqrc_v3/tests/unit/test_correction_stage.py -q`.
- Result: `5 failed, 25 passed, 35 warnings in 3.27s`. All five unsafe states were accepted before
  production changes, reproducing both fresh re-review findings.

### Fix round 2 GREEN

Commit: `bd911ac` (`fix(hqrc-v3): bind correction output identities`)

- Complete validation now reconstructs the canonical output record for every logical key,
  including the current HQRCData pointer/generation files, posterior/checkpoint, and four fixed
  Parquet filenames. The recorded key/path/type/hash map must equal it exactly, so an aliased and
  fully rehashed manifest fails closed before reuse.
- Partial and complete validation now inspect the generation namespace and its current NPZ and
  metadata using `lstat`. Both current files must be real regular files, and the namespace must
  contain exactly those two files; current symlinks plus unknown regular/symlink entries fail
  before downstream cleanup or `COMPLETE` publication and without sampling.
- Valid downstream-boundary checkpoint resume remains zero-sampler and deterministic.
- Focused approval/correction/CLI suite: `42 passed, 30 expected tiny-draw warnings in 3.66s`.
- Full non-slow suite: `431 passed, 8 deselected, 77 expected tiny-draw warnings in 63.64s`.
- Full project Ruff and `git diff --check`: clean. Protected nested lock SHA remains
  `f07f2944707750a9b0753690e6fca2d9c83dbf1d29340e3628483573a2766657`.
- Real paper-source/reduced-PyMC fit plus NetCDF semantic reuse passed in temporary output:
  `1 passed, 36 expected reduced-draw warnings in 13.57s`; protected paper input hashes remained
  unchanged and no paper correction output was written.
- Full paper sampling remains prohibited pending a fresh scoped READY verdict.

## Diagnostic-driven fix round 3 — RED evidence

- Simulated a strict diagnostic rejection that leaves only HQRCData, then retried the same
  profile/RNG seed with independently changed draws, tune, or chains. Coverage includes the real
  paper retry contract from 1,000 to 2,000 retained draws and verifies eventual same-contract
  reuse. Added an exact CLI-help contract for smoke requirements and paper overrides.
- Exact RED command:
  `uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml
  hqrc_v3/tests/unit/test_correction_stage.py
  hqrc_v3/tests/integration/test_cli_corrections.py -q`.
- Result: `5 failed, 33 passed, 30 warnings in 3.67s`: all four sampler-size retries collided with
  the prior partial namespace, and CLI help described the limits as smoke-only.

### Fix round 3 GREEN

Commit: `b254d13` (`fix(hqrc-v3): isolate sampler-size retries`)

- Added the deterministic leaf `draws-<D>-tune-<T>-chains-<C>` below the sampler profile/RNG
  namespace, using the resolved sampler contract. Baseline and sampler seeds remain separate.
- A diagnostic-rejected attempt and a retry changing any one of draws, tune, or chains now occupy
  independent directories. Tests include the paper 1,000-to-2,000 retained-draw retry and prove
  the failed directory is retained untouched while the retry completes; repeating the exact retry
  contract reuses with no additional sampler call. Existing same-contract checkpoint-resume tests
  remain green.
- CLI help and README now state that smoke requires all three explicit limits, while paper accepts
  optional overrides only with at least 1,000 tune/draws and exactly four chains. The README gives
  the intended `--draws 2000 --tune 1000 --chains 4` retry and the full deterministic path layout.
- Focused approval/correction/CLI suite: `47 passed, 30 expected tiny-draw warnings in 4.48s`.
- Full non-slow suite: `436 passed, 8 deselected, 77 expected tiny-draw warnings in 65.97s`.
- Full project Ruff and `git diff --check`: clean. Protected nested lock SHA remains
  `f07f2944707750a9b0753690e6fca2d9c83dbf1d29340e3628483573a2766657`.
- Real paper-source/reduced-PyMC fit plus same-contract NetCDF semantic reuse passed in temporary
  output: `1 passed, 36 expected reduced-draw warnings in 14.21s`; protected inputs were unchanged.
- Diagnostic thresholds were not changed. The failed real paper attempt was not deleted or
  modified, no full fit was run, and no actual paper correction output was published.

## Sampler-geometry fix round 4 — actual XGBoost-B1 diagnosis

Implementation commit: `bd4eb25` (`fix(hqrc-v3): stabilize cyclic hour geometry`)

### Production reconstruction and bounded current/alternative comparison

- Reconstructed the approved XGBoost-B1 context through
  `prepare_causal_correction_inputs(..., profile="smoke")`, using the production source audit,
  residual-manifest validation, and final-baseline validation/reuse path. Exact command:

  ```bash
  HQRC_V3_RUN_DIR=/Users/ijongseung/Documents/GitHub/arima-type/Demadn-Quadratic-Tilting/artifacts/hqrc-v3-paper-20260811 \
  HQRC_V3_APPROVED_AR=/Users/ijongseung/Documents/GitHub/arima-type/Demadn-Quadratic-Tilting/artifacts/hqrc-v3-paper-20260811/ar_diagnostics/xgboost-B1-approved.json \
  uv run --project hqrc_v3 --locked python -c '
  import os
  from pathlib import Path
  import numpy as np
  from hqrc_v3.correction_stage import prepare_causal_correction_inputs
  from hqrc_v3.bayes.model import build_hqrc_model, HQRCModelOptions
  run = Path(os.environ["HQRC_V3_RUN_DIR"])
  approved = Path(os.environ["HQRC_V3_APPROVED_AR"])
  inputs = prepare_causal_correction_inputs(
      run_dir=run,
      config_path=Path("hqrc_v3/configs/experiment.toml").resolve(),
      approved_ar_path=approved,
      profile="smoke",
  )
  print({
      "rows": inputs.hqrc_data.observations.size,
      "events": inputs.hqrc_data.occurrence_ids,
      "obs_min": float(np.min(inputs.hqrc_data.observations)),
      "obs_max": float(np.max(inputs.hqrc_data.observations)),
      "obs_mean": float(np.mean(inputs.hqrc_data.observations)),
      "obs_sd": float(np.std(inputs.hqrc_data.observations)),
      "a": inputs.approved.a,
      "b": inputs.approved.b,
  })
  model = build_hqrc_model(
      inputs.hqrc_data,
      inputs.approved,
      "H3",
      "partial",
      HQRCModelOptions(
          covariance="full",
          include_restriction=True,
          innovation="normal_ar1",
      ),
  )
  print("free_RVs", [(rv.name, tuple(rv.type.shape)) for rv in model.free_RVs])
  print("initial", {
      key: (np.shape(value), float(np.min(value)), float(np.max(value)))
      for key, value in model.initial_point().items()
  })'
  ```

  Result: exact eight ordered occurrences and 1,032 training rows; standardized residual range
  `[-7.8931367543, 4.9646390704]`, mean `-0.7753778602`, SD `1.8157354678`;
  approved context Beta parameters `a=36.7254702661`, `b=3.2745297339`. The current free
  hour coordinates were `sigma_gamma (2,)` plus centered `gamma_innovation (2,23)`.
- After that production reconstruction, every sampler diagnostic read the immutable HQRCData
  generation from the failed attempt and wrote draws/JSON/NetCDF only below `/tmp`. The common
  command form was:

  ```bash
  uv run --project hqrc_v3 --locked python /tmp/hqrc_xgb_diag.py \
    <current-or-adapt_diag> /tmp/<diagnostic>.json <draws> <tune>
  ```

  The helper called the actual H3/full-covariance/partial-pooling/restriction/Gaussian-AR(1)
  PyMC model with sampler seed `20260811`, four serial chains, and `target_accept=0.99`, retained
  raw draws before applying any production gate, and emitted elementwise ArviZ diagnostics plus
  chain-level divergence/BFMI/energy/step-size summaries.
- Current `jitter+adapt_diag`, 150 draws/250 tune: 143 seconds, max R-hat `1.25`, minimum
  bulk/tail ESS `13/14`, zero divergences, and a `quadpotential.py` overflow. The worst element
  was `between_cov_0_stds[2]` / `between_scale[0,2]` at R-hat `1.25`, bulk ESS `13`; its chain
  means were `0.460560/0.197637/0.221174/0.192907`. `sigma_gamma[0]` had bulk ESS `69`; its
  chain means were `0.124445/0.128405/0.115634/0.098457`. Chain BFMI was
  `0.712/0.598/0.640/0.689`, and energy means were
  `976.470/976.673/977.414/972.747`.
- Deterministic `adapt_diag`, still centered, 150 draws/250 tune: 111 seconds, max R-hat `1.28`,
  minimum bulk/tail ESS `12/14`, zero divergences. The failure moved directly onto
  `sigma_gamma[0]` (R-hat `1.28`, bulk/tail ESS `12/14`) and `gamma[0,13]` (R-hat `1.22`, bulk
  ESS `14`); `gamma_innovation[0,16]` had bulk ESS `21`. `sigma_gamma[0]` chain means were
  `0.076549/0.120440/0.114331/0.117304`, with chain 0 reaching `0.003932`. Chain 0 BFMI was
  `0.178`, versus `0.573/0.637/0.829` for chains 1--3; energy means were
  `958.528/976.615/974.723/975.689`. Thus stable initialization alone did not repair the hour
  scale/innovation funnel.

### RED evidence and statistical correction

- First RED command:

  ```bash
  uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml \
    hqrc_v3/tests/unit/test_hqrc_model.py::test_h3_noncenters_cyclic_hour_random_walk_without_changing_output_shape \
    hqrc_v3/tests/integration/test_hqrc_sampling.py::test_pymc_sampler_uses_fixed_stable_initialization_and_records_geometry -q
  ```

  Result: `2 failed in 1.30s`; `gamma_innovation_raw` was absent and PyMC received no explicit
  `init` argument.
- An exact non-centered change of variables without correcting the scale-dependent closure
  normalization made the defect explicit: the same 150/250 run produced max R-hat `1.87`,
  bulk/tail ESS `6/19`, and `143` divergences split `61/0/0/82` across chains. This showed that
  the prior closure term, not merely random initialization, was driving the zero-scale boundary.
- The specified model declares `sigma_gamma ~ HalfNormal(0.5)` and a normalized cyclic RW1.
  With 23 free increments, the old closure potential contributed an uncancelled `1/sigma_gamma`
  normalization factor, silently distorting that declared scale prior and producing an improper
  zero-boundary funnel. Direct RED command:

  ```bash
  uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml \
    hqrc_v3/tests/unit/test_hqrc_model.py::test_noncentered_cyclic_closure_does_not_change_halfnormal_scale_prior -q
  ```

  Result: `1 failed in 2.85s`; the closure-only log densities at narrow and wide scale vectors
  differed by `4.24849524`.
- Implemented the exact normalized non-centered coordinate transform: raw standard-Normal
  `gamma_innovation_raw (2,23)`, deterministic scaled `gamma_innovation (2,23)`, the unchanged
  centered `gamma (2,24)` output, and the required `+log(sigma_gamma)` closure normalization.
  Algebraic tests prove equality to the centered cyclic RW1 density after its coordinate
  Jacobian and prove that the closure no longer modifies the declared HalfNormal scale prior.
  H3, full covariance, partial pooling, restriction, every named prior scale/shape, the approved
  Beta calibration, posterior-estimated phi, and event-reset Gaussian AR(1) likelihood remain
  unchanged.
- Fixed production PyMC initialization to `adapt_diag`. Initialization and geometry are recorded
  in `hqrc_sampler_json`, `hqrc_model_json`, checkpoint/manifest identity, and the deterministic
  namespace
  `init-adapt_diag-geometry-noncentered-cyclic-hour-rw1-v1/`. This leaves both failed real paper
  directories untouched and prevents their partial HQRCData generations from colliding with a
  corrected run. `target_accept` remains `0.99` for paper and all diagnostic thresholds remain
  unchanged.

### Corrected real XGBoost-B1 four-chain evidence

- Bounded 150 draws/250 tune: 83 seconds, max R-hat `1.05`, minimum bulk/tail ESS `67/93`, zero
  divergences, BFMI `0.959/0.747/0.770/0.854`. This removed the hour-funnel failure; the remaining
  worst elements were weak between-event covariance scales under the deliberately short draw
  count.
- Required larger bounded validation command:

  ```bash
  uv run --project hqrc_v3 --locked python /tmp/hqrc_xgb_diag.py \
    adapt_diag /tmp/hqrc-xgb-normalized-noncentered-500.json 500 500
  ```

  Result: 417 seconds, four chains, max R-hat `1.02`, minimum bulk/tail ESS `317/228`, zero
  divergences, BFMI `0.662/0.669/0.795/0.799`. Chain means demonstrate common geometry:
  `sigma_gamma[0] = 0.131642/0.132454/0.135680/0.131875`,
  `sigma_gamma[1] = 0.377320/0.389996/0.371932/0.372924`,
  `sigma_r = 0.563278/0.563069/0.561968/0.562798`, and
  `phi = 0.910147/0.909946/0.908552/0.909663`. Scaled `gamma_innovation` aggregate chain means
  were `-0.022485/-0.022781/-0.022730/-0.022688`. The worst remaining element was the restriction
  quadratic coefficient `delta[0,2]` at R-hat `1.02`, bulk/tail ESS `317/228`; hour raw-coordinate
  ESS values were mostly above 1,000 and `sigma_gamma[1]` bulk/tail ESS was `478/951`.
- This bounded run demonstrates all four chains on the same finite-energy geometry without
  divergences. It is not represented as passing the 2,000-draw paper publication gate; the
  controller retains ownership of that prohibited full fit after fresh review.

### Fix round 4 verification

- Focused Bayesian/approval/correction/CLI command:

  ```bash
  uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml \
    hqrc_v3/tests/integration/test_ar_artifact.py \
    hqrc_v3/tests/unit/test_hqrc_model.py \
    hqrc_v3/tests/integration/test_hqrc_sampling.py \
    hqrc_v3/tests/unit/test_correction_stage.py \
    hqrc_v3/tests/integration/test_cli_corrections.py -q
  ```

  Result: `85 passed, 56 expected tiny-draw warnings in 13.32s`.
- Full non-slow command:

  ```bash
  uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml \
    hqrc_v3/tests -m 'not slow' -q
  ```

  Result: `442 passed, 8 deselected, 77 expected tiny-draw warnings in 66.90s`.
- Real paper-source XGBoost-B1 reduced-PyMC fit plus NetCDF semantic reuse command:

  ```bash
  HQRC_V3_REAL_RUN_DIR=/Users/ijongseung/Documents/GitHub/arima-type/Demadn-Quadratic-Tilting/artifacts/hqrc-v3-paper-20260811 \
  HQRC_V3_REAL_APPROVED_AR=/Users/ijongseung/Documents/GitHub/arima-type/Demadn-Quadratic-Tilting/artifacts/hqrc-v3-paper-20260811/ar_diagnostics/xgboost-B1-approved.json \
  uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml \
    hqrc_v3/tests/slow/test_real_correction_stage.py -m slow -q
  ```

  Result: `1 passed, 36 expected reduced-draw warnings in 14.28s`; the test hashes protected
  baseline/residual/approval inputs before and after and writes correction output only below the
  pytest temporary directory.
- `uv run --project hqrc_v3 --locked ruff check --config hqrc_v3/pyproject.toml
  hqrc_v3/src hqrc_v3/tests` — `All checks passed!`.
- `git diff --check` and staged `git diff --cached --check` — clean.
- Protected untracked `hqrc_v3/uv.lock` remains unstaged and byte-identical at
  `f07f2944707750a9b0753690e6fca2d9c83dbf1d29340e3628483573a2766657`.
- No existing real correction artifact was deleted or modified, no diagnostic threshold or
  `target_accept` was changed, and no final 2,000-draw paper fit was run.
