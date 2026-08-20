# Paper-profile divergence retry specification

## Goal

Keep the paper diagnostic gate at zero divergences while automatically retrying a fold once when divergences are the only failed diagnostic.

## Required behavior

- The first paper attempt remains unchanged: `target_accept=0.99`, at least `tune=1000`, at least `draws=1000`, and exactly four chains.
- A retry is eligible only when the sampler raised a structured diagnostic failure with `divergences > 0`, `max_rhat <= 1.01`, `min_bulk_ess >= 400`, and `min_tail_ess >= 400`.
- The retry runs once with `target_accept=0.999`, `tune=max(2000, base_tune)`, unchanged draws/chains/cores/backend/device/init, and a deterministic retry seed derived from the root seed, variant, held-out occurrence, and retry attempt.
- Smoke runs and non-diagnostic failures never retry. A failed retry remains a hard failure; no divergent posterior is published.
- Existing successful base publications keep their exact identities and are reused without refitting.
- A successful retry uses a separate immutable identity. Its signed manifest records the base sampler digest and the structured first-attempt diagnostics.
- Resume prefers a valid base result, otherwise reuses a valid retry result before attempting new sampling.
- H1, H2, and H3 aggregates may contain a mixture of valid base and retry folds. Each fold's manifest and sampler remain explicit in aggregate provenance.
- The implementation is pure Python and preserves Windows, Linux, and macOS execution behavior.

## Operational boundary

Code loaded by already-running Python processes is not replaced. The current run remains untouched; the new behavior applies when affected contexts are launched again.
