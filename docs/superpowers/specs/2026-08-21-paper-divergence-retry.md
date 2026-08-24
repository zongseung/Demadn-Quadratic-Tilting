# Paper-profile diagnostic rescue specification

## Goal

Keep the strict paper diagnostic gate while automatically rescuing one finite diagnostic failure per fold without changing the statistical model.

## Required behavior

- The first paper attempt remains unchanged: `target_accept=0.99`, at least `tune=1000`, at least `draws=1000`, and exactly four chains.
- A rescue is eligible only when the sampler raised a structured, finite diagnostic failure: `divergences > 0`, `max_rhat > 1.01`, `min_bulk_ess < 400`, or `min_tail_ess < 400`.
- The rescue runs once with `target_accept=0.999`, `tune=max(3000, base_tune)`, `draws=max(2000, base_draws)`, unchanged chains/cores/backend/device/init, and a deterministic retry seed derived from the root seed, variant, held-out occurrence, and retry attempt.
- Pyro rescue fits use `full_mass=True`; base fits retain their existing diagonal adaptation and exact identity.
- Smoke runs, missing/malformed diagnostics, and non-diagnostic failures never rescue. A failed rescue remains a hard failure; no posterior that fails the gate is published.
- Existing successful base publications keep their exact identities and are reused without refitting.
- Existing successful divergence-only retry publications remain valid read-only resume candidates; new failures never sample that superseded contract.
- A successful rescue uses a separate immutable identity. Its signed manifest records the base sampler digest and the complete structured first-attempt diagnostics.
- Resume prefers a valid base result, otherwise reuses a valid rescue result before attempting new sampling.
- H1, H2, and H3 aggregates may contain a mixture of valid base and rescue folds. Each fold's manifest and sampler remain explicit in aggregate provenance.
- The implementation is pure Python and preserves Windows, Linux, and macOS execution behavior.

## Operational boundary

Code loaded by already-running Python processes is not replaced. The current run remains untouched; the new behavior applies when affected contexts are launched again.
