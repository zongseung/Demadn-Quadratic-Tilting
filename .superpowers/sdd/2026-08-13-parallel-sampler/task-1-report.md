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

Commit: pending
