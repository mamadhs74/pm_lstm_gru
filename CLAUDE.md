# Project context

Residual TCN pipeline correcting a causal EMA reference. Target is a
tyre-related particle count signal (P_VA), predicted 1 s ahead at 5 Hz from
six vehicle-dynamics features (speed, longitudinal/lateral acceleration,
steering rate, delta-v, jerk).

Data: one recording, one vehicle, one session. This is a hard ceiling on
what the results can support; no code change fixes it.

## Current status

**Evaluation split bug — found, fixed, committed.**
`src/evaluate.py` built its evaluation frame from train + val + test
concatenated instead of the held-out test split alone, so reported metrics
were computed partly over training data. Introduced 2026-04-30 by commit
`463c9a6` (a change meant only to export full-sequence predictions for
AP6/AP7 scenario analysis; the concatenated frame was also reused for the
headline metrics). Fixed by restoring `splits[split_key]` as the sole
evaluation frame, committed in `1fdede0`.

**Inverse-transform consistency check — fixed.**
The direct-mode check compared absolute reconstruction error against a
fixed `1e-5` bound. `WindowedDataset` casts targets to float32 before the
`target_scaler` round-trip; at P_VA's scale (~1e3) one float32 ULP is
~2.4e-4, so the check failed on rounding alone, with no real
inconsistency, and blocked all direct-mode evaluation. Fixed in `0999081`:
both `direct` and `residual_shift` modes now use `np.allclose(rtol=1e-5,
atol=1e-3)`, with a separate, mode-appropriate reconstruction for each
(direct compares the inverse-transformed target directly to physical
truth; residual_shift reconstructs physical truth from the true residual
via the EMA formula before comparing — the two are not interchangeable,
since the residual-mode inverse-transform yields a residual ratio, not a
physical value). Direct-mode evaluation now runs end to end.

**Direct-mode comparison — complete.** Five seeds (10/20/30/40/50), both
evaluation scopes (active, stationary). Direct TCN loses to both trivial
baselines (persistence, causal EMA) and to the residual-shift model in
every scope:

| scope | model | MAE mean ± std (5 seeds) |
|---|---|---|
| active | persistence (baseline) | 260.49 |
| active | causal EMA (baseline) | 339.62 |
| active | direct TCN | 719.68 ± 216.45 |
| active | residual-shift TCN | 615.56 ± 71.07 |
| stationary | persistence (baseline) | 10.35 |
| stationary | causal EMA (baseline) | 8.57 |
| stationary | direct TCN | 38.95 ± 35.75 |
| stationary | residual-shift TCN | 8.78 ± 0.19 |

**seed-20 residual_shift checkpoint — overwritten, recovered.**
`.runs/tcn_residual_shift_seed20_best.pt` and the corresponding
`publication_results/training/residual_shift/seed_20/` artifacts were
overwritten on 2026-07-27 by an accidental short training run. The
converged checkpoint and training artifacts behind the reported seed-20
result (active-scope MAE 541.761374, part of the 615.56 ± 71.07 aggregate
above) are not recoverable in `.runs/` or the live `publication_results/`
training tree, but are preserved in `ieee_publication_evidence.zip` and
have been restored from there into `evidence_bundle_2026-07-28.zip`. The
live `.runs/tcn_residual_shift_seed20_best.pt` on disk does not match
either bundle and should not be used for anything.

## Evidence bundles

- **`ieee_publication_evidence.zip`** — untouched snapshot from
  2026-07-23. Ground truth for the original 5-seed residual_shift results.
  Never modified after creation.
- **`evidence_bundle_2026-07-28.zip`** — current bundle. Adds the
  completed direct-mode results (both scopes, five seeds), restores the
  seed-20 residual_shift training artifacts from the 2026-07-23 snapshot
  (superseding the 2026-07-27 accidental overwrite), and refreshes the
  `evidence/*.py` code snapshot to the commit that actually produced the
  direct-mode results (`0999081`). Includes a sha256 manifest and a
  `README.txt` documenting exactly what was restored/refreshed and why.

## Answered questions

1. **Do the EMI-V final-report figures (§3.2) predate or postdate the
   evaluation split-bug fix?** Predate. The report is dated 2026-04-24;
   the split bug was introduced 2026-04-30 (`463c9a6`). The report's
   figures are unaffected by the bug — they were never processed through
   the concatenated-frame code path.
2. **Why did the inverse-transform consistency check fail?** The
   tolerance defect described above (absolute `1e-5` bound too tight for
   a float32 round-trip at P_VA's physical scale), not a data or model
   bug.

## Still open

- **P_VA channel definition** — instrument, inlet, particle size range —
  is unconfirmed.
- **Ring-test data access** (ILS TF3, UNECE PMP) is unknown.

## Evaluation rules

- Chronological splits only, with purge zones at boundaries. Never random.
- Compare against persistence and EMA before claiming any model benefit.
- Report multi-seed spread, never single values.
- Treat data filtering as part of the method, documented, not as
  preprocessing.
- State the target channel's instrument and size range explicitly, or
  state that it is unconfirmed.
- Output paths must encode configuration and run identity (target_mode,
  seed, scope) — never a shared path that a different config can silently
  overwrite.
- Smoke tests must never share a path with production runs. A smoke-test
  checkpoint or result written to the same path as a real run is a
  standing risk of silent overwrite, not a hypothetical one.
- Tolerances in consistency checks must be relative, not absolute, when
  the checked quantity's scale can vary (e.g. across target_mode or
  target column).
- A consistency check must exist on every code path it's meant to cover,
  not just the path it happened to be written for.
