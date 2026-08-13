# Task 1 report — parallel PyMC-chain sampler contract

## Implementation

- Added `cores` to `sample_hqrc`, defaulting to `1`, with strict positive-integer and `cores <= chains` validation. The value is passed to PyMC and stored in `hqrc_sampler_json`.
- Threaded the resolved core count through LOEO fold/primary APIs, identity, and posterior provenance validation.
- Threaded it through the causal correction sampler contract/API, output namespace, identity, and posterior provenance validation.
- Updated sampler test doubles and focused coverage for defaults, explicit multi-core propagation, invalid values, and identity binding.

## Verification

- `uv run --project hqrc_v3 pytest -q hqrc_v3/tests/integration/test_hqrc_sampling.py --disable-warnings` — 5 passed.
- `uv run --project hqrc_v3 pytest -q hqrc_v3/tests/unit/test_loeo_stage.py -k 'sampler or strict_diagnostic' --disable-warnings` — 2 passed.
- `uv run --project hqrc_v3 pytest -q hqrc_v3/tests/unit/test_correction_stage.py -k 'sampler or retry' --disable-warnings` — 5 passed.
- `uv run --project hqrc_v3 pytest -q hqrc_v3/tests/unit/test_loeo_stage.py -k sampler_cores --disable-warnings` — 1 passed.
- `uv run --project hqrc_v3 pytest -q hqrc_v3/tests/unit/test_correction_stage.py -k causal_sampler_contract --disable-warnings` — 1 passed.
- Ruff passed on all touched source and test files.

## Concerns

No sampler or paper run was started. `hqrc_v3/uv.lock` was already untracked and was deliberately left untouched; no artifacts were edited.

Commit: `0660e0973a0e544b007d99dd86da9f3721b845dd` (`Add explicit HQRC sampler core contract`)

## Fix round 1

- The fresh-process benchmark request now writes, validates, and digest-binds a canonical `cores` field (default `1`), rejecting boolean, non-positive, and greater-than-chain values; both PyMC and nutpie worker requests forward it to `sample_hqrc`.
- The causal correction CLI exposes `--cores` and forwards both omitted and explicit values.
- LOEO-primary translates invalid lower-level core contracts to `LOEOPrimaryError`.

Verification:

- `uv run --project hqrc_v3 pytest -q hqrc_v3/tests/integration/test_sampler_worker.py hqrc_v3/tests/integration/test_cli_corrections.py hqrc_v3/tests/unit/test_loeo_primary.py -k 'cores or core_contract or cli or parent_launches or boolean_version' --disable-warnings` — 16 passed, 37 deselected.
- Ruff passed on all fix-round source and tests.

No real sampler run started; artifacts and `hqrc_v3/uv.lock` remain untouched.
