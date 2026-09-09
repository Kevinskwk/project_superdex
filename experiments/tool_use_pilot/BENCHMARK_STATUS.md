# Benchmark status — 9 September 2026

**All 18 retained isolated variants now have full nominal feasibility passes,
plus the insert–turn–hold composite. This is not yet a frozen or robustness-qualified benchmark.** The current code
uses the curved gel, force-only model observations, coupled jaws, explicit
impedance, turn-and-hold key endpoints and diagnostic-only wrench matching.
The bounded 9 September campaign reran the current tasks, replaced flat peeling
with blade-first cylinder contact, and tested targeted repairs. See
[the verification record](VERIFICATION_20260909.md) and its linked all-attempt
report. Historical endpoint failures are not retrospectively converted into
successes: the new turn-and-hold results are newly simulated full rollouts.

Original bounded result: **17/19 isolated variants have accepted full nominal
passes**, plus the insert–turn–hold composite at two spring settings. Convex
scraping remains under contact-model review. Curved-slot following reaches its
travel/contact goals but its 30.67 s recording is censored at 22.72 s in release;
it is not an accepted full episode. There are 45 retained attempts, including
11 runtime-censored recordings; this mixed repair pool is not a success-rate
estimate. No large-scale collection was launched.

Follow-up user decision and completed rerun: convex scraping is removed from the active benchmark;
only flat scraping remains. Historical convex recordings are preserved and
explicitly marked retired in the derived index. The active set now has 18
isolated variants. Curved-slot #45 completed all 30.67 s / 3067 frames in
1815.6 s wall time, with unchanged mechanics and an exact shared-prefix match
to #41. Its task and physical checks pass. The ledger now contains 46 recordings,
11 historical runtime-censored attempts, zero setup errors and zero unresolved
reservations; these mixed attempts are not a pooled success-rate estimate.

## Latest evidence by task

| Task | Evidence | Current status |
|---|---|---|
| Flat scraping / drawing | Current #0/#1: 18.33/18.34 mm actual follow travel, 100% contact, complete | Nominal passes |
| Cylinder peeler following | #36/#39: 21.35/21.37 mm of 25 mm, blade-first, 100% blade contact, zero holder contact, complete release | Pass with SDF and mesh; rigid following, not material removal |
| Concave drawing | #29/#38: 18.47/18.41 mm of 20 mm after tip-pivot compensation, 100% contact | Nominal passes; reduced previous 27.32 mm overshoot |
| Straight wall/slot | #22/#30: 20.47/20.49 mm, 100% guide contact at 45° / 0.5 N | Repaired nominal passes |
| Rounded wall | #40: 23.59 mm of 20 mm stroke, 100% guide contact, complete | Repaired nominal pass; characterize overshoot under perturbations |
| Curved slot | #45: full 30.67 s; 23.59 mm tip travel, 100% guide contact; sampled all-environment overlap ≤0.098 mm | Full nominal pass; final net contact <0.001 N; retention maintained |
| Normal / mirrored hook | #42/#43: 42.75/42.74 mm of 40 mm goal, retained; ~12.3° rotation, both full 32 s runs | Nominal passes; no broad grasp-robustness claim |
| Round/square/hex/D/key insertion | #10–14 all complete and pass with current controller | Five nominal feasibility passes |
| Spring/friction/detent turning | #15–17 all complete, hold seated at 91.51°/92.24°/92.38° | All new endpoints pass; native-friction #33/#34 also pass with smaller transients |
| Insert-and-turn composite | #25/#26 both complete; ~16 mm seated, 91.53°/92.11° final rotation | New endpoint passes; blocked/misaligned handoffs still need current-version trials |

Current evidence: `output/task_verification_20260909/report.json` and immutable
per-phase source snapshots. Earlier `output/mechanics_repair/mechanics_summary.json`
and `output/curved_gel_pilot/report.json` remain historical evidence, not current
qualification. The campaigns contain mixed configurations, repeated deterministic
controls, deliberate negatives and debugging failures. A pooled success rate
across them would be misleading. Some older surface success flags predate the
working-tip travel audit and must not be used as current qualification.

## What passing means

- **Episode accepted for world-model use:** physically valid, complete, with
  no unresolved contact-geometry issue. A valid task failure may be accepted.
- **Successful demonstration:** accepted episode plus measured task success.
- **Variant qualified:** repeated nominal success on one frozen configuration,
  followed by a defined perturbation envelope. Three identical deterministic
  repeats establish reproducibility, not independent robustness evidence.
- Wrench RMSE, correlation and force/torque reconstruction are diagnostics,
  never selection gates. Numerical blow-ups, NaNs, inversion, gross penetration,
  loss of grasp or safety violations are not excused by that decision.
- Deliberate misalignment/miss cases should not all succeed. They need correct
  outcome labels, plausible mechanics and appropriate recovery/abort behaviour.

## Remaining before a benchmark freeze

1. Freeze the retained revision-18 task settings and tested endpoints from this
   campaign; convex scraping is already retired. All retained nominal variants
   now have completion evidence. Historical timeouts remain in the ledger.
   Keep matched flat-reference episodes separate from primary curved-gel trials.
2. Establish repeated current-version insertion/turning prerequisite evidence.
   The two composite checks used a read-only older prerequisite set for diagnostic
   authorization; they demonstrate mechanics, not frozen-version qualification.
   Do not relax task criteria or tune gel hardness merely to force nominal success.
3. For retained variants, run a small independently perturbed set (e.g. 3–5
   configurations initially): orientation/offset, clearance, grip and gel
   parameters, surface friction, camera view. Include plausible near misses,
   blocked cases and safe failure/recovery. Establish and document parameter
   bounds; expand sample counts before claiming statistical robustness.
4. Review actual RGB at start/contact/peak-load/end and independent geometry
   checks. Assess timestep/solver sensitivity of trajectories, contact events
   and force fields—not just wrench matching. The previous four-iteration
   friction solves were not converged; disabling a wrench gate does not cure
   that limitation.
5. Freeze HDF5/model-input contracts, units, frames, assets/hashes, seeded splits,
   acceptance masks and task-success definitions. Check loaders, reproducibility
   and branch-state restoration on the release candidate. Keep contact truth,
   dense/corrected wrenches and cell moments out of model input.
6. Recheck vision-only versus force-field-only versus vision+force-field under
   matched noisy/occluded point clouds, with held-out physical conditions and
   tactile ablations. Scripted collection still uses privileged force feedback;
   it is not evidence of learned tactile control or cross-domain transfer.

## Completion boundary

A simulation benchmark v1 is complete when a declared task set and perturbation
envelope have reproducible mechanics, validated labels and observations, fixed
splits/metrics, runnable evaluation and representative reviewed recordings.
Optional difficult guide variants can remain explicitly experimental; they
must not be silently counted as part of a completed core. Large-scale collection
should follow this freeze, not precede it.

Real-gel calibration and real-system evaluation remain necessary for claims
of sensor fidelity or sim-to-real transfer. ACWM training/generalization and
uncertainty evaluation on the main server are a separate project milestone,
not prerequisites to finishing the simulation task package itself.
