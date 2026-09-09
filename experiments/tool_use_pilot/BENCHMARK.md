# Contact-rich tool-use diversity pilot

This is a small mechanics and observability pilot for ACWM, not a finished
benchmark or evidence of sim-to-real transfer. It retains the existing Franka,
Franka gripper and two compliant GelSight Mini assemblies. Physics uses FP32;
independent episodes use CPU processes. EGL rendering and the frozen
Point-M2AE probe use the GPU.

Current status and completion checklist: [BENCHMARK_STATUS.md](BENCHMARK_STATUS.md).

## Current gel and acceptance policy (9 September)

`source_surface` is the primary/default curved gel. `matched_box` is the
explicit flat reference: identical XY topology and 2 mm marker spacing, with
marker locations on nodes. Both meshes include additional shoulder nodes;
neither promises exact torque from the 63 force vectors alone. `legacy_box`
is retained only for historical replay (its marker pitch is different).
Use `BenchmarkSpec(gel_geometry='matched_box')` for the flat reference; do not
silently pool that reference domain with primary curved-gel evaluations.

Tactile-derived versus contact-wrench agreement is **diagnostic only**. It
never rejects an episode or gates nominal task qualification. New dataset
indices use schema v2: `episode_accepted`/`world_model_eligible` require valid
mechanics, a complete rollout, and no unresolved contact-geometry review;
`imitation_eligible` additionally requires task success. Valid task failures
remain useful world-model examples. The old `tactile_valid` name is retained
for compatibility as a diagnostic, not an eligibility mask. The former
`compact_wrench_eligible`/`dense_wrench_eligible` outputs are replaced by
`*_wrench_within_reference` diagnostics. Old HDF5 and index files are not
silently rewritten; regenerate an index to apply the new policy.

Model tactile input remains the two 7x9x3 force fields, without intrinsic cell
moments or corrected/dense/contact wrenches. Safety/contact-depth/retention and
data-integrity checks are not removed by this policy.

Revision 12 defaults to the source-fitted curved gel with 2 mm XY marker pitch.
The config records `gel_geometry`, HDF5 records the mesh SHA-256, and each phase
snapshots the gel code and mesh. Revision ≤11 manifests without that field
resolve to `legacy_box` for historical replay. The compact field uses nearest-
marker force bins; `dense_nodal_gel_wrench` and
`dense_nodal_inferred_extrinsic_wrench` retain unbinned moment accounting.

## Task structure

| Family | Implemented variants | Observable outcome / intended uncertainty |
|---|---|---|
| Surface following | Flat scrape/draw; cylindrical peeling-style stroke; concave draw; straight/rounded wall; straight/curved slot | Contact continuity, working-tip progress, blade versus holder contact, lateral error; contact loss versus resistance and guide engagement |
| Capture and pull | Normal/mirrored hook, working-end and crossbar dimensions | Capture, pull, unload and recovery; engagement versus miss |
| Insertion | Round, square, hexagonal, D and key cross-sections | Measured seating, blocked approach, withdrawal; clearance and rotational alignment |
| Turning | Prepared-seated key with passive spring, friction or detent resistance | Turn and hold; hidden load mode/resistance |
| Composite | Insert key, turn and hold seated | Continuous transition after measured seating; no reset or teleport between stages |

The active catalog has 18 isolated variants plus the composite. Scraping is
**flat-only**: convex scraping was retired by user decision after the 9 September
repair sweep. Canonical, screening, qualification, load-sweep, diversity and
randomized recipes do not generate it. Historical explicit manifests and results
remain readable/replayable, but are excluded from active benchmark eligibility.
Concave drawing, convex-cylinder peeling and wall/slot following remain in scope;
they are not convex scraping.

Drawing has no ink; peeling/scraping have no material removal or cutting. These
are **rigid contact variants**, not simulations of those material processes.
SCFields meshes are external, ignored assets. Their collision proxy and source
are recorded. The current cylindrical peeler uses the watertight
`peeler_1_head_rectangular_handle_h0_96_hnd0_99` asset with its opening preserved;
some other peeler assets have convex proxies and are not equivalent substitutes.
Revision 17 replaces flat peeling in the canonical recipe with a 40 mm-diameter,
120 mm-long cylinder whose axis follows the stroke. The target crown is aligned
to the central blade edge, not the lower holder ends. Blade-first contact,
at least 80% blade-contact continuity, at most 5% holder-contact frames, and
the existing tip-travel/retention checks are required. Historical `flat_peel`
manifests remain replayable but are not the recommended peeling task.

Revision 18 incorporates the bounded 9 September repair experiments:
blade-matched cylinder peeling at 1 N with a 0.5 mm SDF grid; wall/slot heading
45 degrees at 0.5 N floor load; working-tip XY rotation compensation on curved
surfaces; and native joint friction for friction-loaded turning. The curved
height surface is still the same closed solid, with two lateral mesh samples
instead of redundant coplanar rows. Convex scraping's 0.5 N setting is a better
motion candidate, **not contact-model qualification**. Historical revisions
retain their old default values. The flat matched gel remains a reference.

The working-tip audit replaces, rather than supplements, the old root-travel
gate: a rotating tool can follow the requested tip path with much less root
translation. Contact continuity, lateral error, minimum tip travel and all
physical safety limits are unchanged. Recorded old scores remain available in
HDF5; the report independently recomputes tip travel and derived acceptance.

The benchmark separates geometry, material/contact parameters, motion and
observation corruption. Implemented parameters include scale, profile,
clearance, chamfer, curvature, fixture translation/rotation, wrist roll/pitch,
grasp depth, speed, grip force, gel stiffness, grip/environment friction and
camera pose. A parameter being implemented does not mean its range is qualified.
The two combined geometry configurations are not one-factor causal tests.
Fixture perturbations are relative to a nominal controller path; this is not
a general geodesic-following policy for arbitrary rotated surfaces.
Prepared turning must keep the key aligned with its socket at initialization;
the current random recipe varies whole-assembly wrist posture and load but not
relative socket misalignment. Earlier unrestricted random manifests remain as
setup stress tests, not a qualified seated-turning distribution.

## Qualification and causal controls

- Prepared grasps only; the initial tool constraint is released at 1 s.
  Wrist pre-positioning, if requested, happens before the tool is created.
- Record the whole configured task, including its final hold. Task progress
  cannot override grasp drift, penetration, gel inversion or force/torque limits.
- Local qualification requires at least three nominal trials and all nominal
  trials in that cell to pass mechanics and task success. A single screen is
  only feasibility evidence. Repeated deterministic trials are not independent
  statistical evidence.
- Wall/slot following additionally requires actual guide contact; contact with
  the floor alone is insufficient. Earlier files without this channel cannot
  establish guide-following qualification.
- Composite execution is gated on three full nominal insertion and turning
  successes. Seating is measured from actual working-end vertices and must dwell
  before the turning controller starts. The socket is passively loaded.
  Here nominal means aligned geometry (≤0.15 mm lateral offset, no fixture/wrist
  pose perturbation), including older recipes named `aligned` or `captured` and
  aligned resistance probes. `gate_evidence.json` identifies the exact episodes;
  near misses and incomplete configured endpoints cannot satisfy the gate.
- Branch experiments restore the same serialized physical and collector state.
  A repeated 20-step continuation must match exactly; sibling prefixes are
  hashed. Anchors are selected from successful screening episodes, then rerun
  with the recorded branch-phase implementation. This does not claim an exact
  replay of an anchor generated by an older implementation.
- Revision 15+ disables ongoing tool-pose feedback by default; surface collection
  still uses privileged precontact calibration and contact-force control.
  These are oracle channels, not vision/tactile policy inputs. Do not
  interpret scripted success as demonstrated tactile policy benefit.

Surface audits require ≥80% contact continuity and path progress, with <3 mm
p95 lateral error. Existing full-episode safety limits remain unchanged; see
`decisions.stop_reason`, `pilot.metrics` and `key_stages.stage_audit` for the
authoritative thresholds. Runtime-budget truncation is reported separately:
it invalidates a full-rollout claim but is not itself a physical defect.
Working-tip travel is independently checked during the follow phase, not after
release; it is not a material-removal measure. Surface revision 11 compensates the
modeled working-edge height during attack-angle changes. Final return/withdrawal
checks apply to historical return tasks and standalone insertion. Revision 16
turning/composite instead require the final 0.3 s seated at 90 +/-5 degrees;
neither reverse rotation nor withdrawal is required.

Mochi combines collider friction coefficients by their geometric mean.
With `effective_friction=true`, the environment actor receives
`friction**2 / gel_friction`, so the desired environment/tool coefficient is
independent of gel/tool friction. Legacy false mode preserves actor-coefficient
semantics. The HDF5 stores both requested and effective coefficients.

An intermediate key-family helper also changed the tool's coefficient. For
affected snapshots (including the initial sensing block), actual grip friction
is √0.7 ≈0.837 and external friction is 0.5, not the initially reported 1.4 and
√0.7. The reporter/probe correct the derived HDF5 attributes, retain their
`original_reported_*` values and add `audited_friction_json`. Signals/configs
are unchanged. The targeted probe remains a constant-friction comparison;
affected one-factor friction sweeps are **not** isolated-factor evidence.
The current implementation applies compensation only to the socket.

## Data and sensing analysis

HDF5 extends the previous ACWM-compatible recorder: actions, timestamps, object
and environment poses, mesh geometry, camera calibration, both 7×9×3 tactile
fields and surface coordinates, contact truth, inferred wrench and phase/outcome
channels. Point clouds are reconstructed from stored meshes and poses; no dense
per-frame point-cloud dataset is required. Oracle channels remain explicitly
identified. Configurations, seeds, source snapshots and failed attempts are kept.

Wrench comparisons use world axes about the instantaneous tool COM, with
Newton–Euler compensation. Rotor torque additionally transports moments to the
rotor origin and projects onto its axis. Same-engine contact agreement checks
bookkeeping and local consistency, not independent real sensor calibration.

Independent geometry checking found that coarse broad scraper faces can overlap
the surface more deeply than reported contact points indicate. The reporter now
checks every recorded surface frame against the saved height mesh using 2 mm
tool-surface sampling, revoking validity for sustained overlap above 1 mm.
Collision tessellation is repaired separately. **Use the final report's validity
mask in `dataset_index.json`, not the older HDF5 `physical_valid` attribute alone.** Original recorder
metrics are retained for audit; wall/slot side intersections additionally need
the general mesh check. A replay matching numerically cannot certify geometry.
The default training index also excludes flat-contact cases flagged by the
force-direction review: large apparent tangential/normal ratios need an
edge/contact-manifold explanation and cannot be interpreted as calibrated
frictional shear. The planar diagnostic records oblique-normal load and
per-contact Coulomb excess separately; these are oracle diagnostics, not inputs.

The first long counterfactual continuations were capped; completed recordings
and checkpoints remain, and cancellations consume the trajectory ledger.
Current branch runs use an explicit six-second response window by default,
with a per-branch runtime limit. They test local action response, not successful
completion of a full return/recovery sequence. A legacy identical-action group
is retained as a replay control and is not action-causality evidence.

The targeted turning probe uses 12 physical conditions: three grip forces ×
four resistances. All must pass full-rollout qualification before model fitting.
Frozen Point-M2AE features use clean, noisy, randomly occluded and contact-region
masked point clouds. Pre-turn/during-turn windows, held-out grip groups,
preload/shuffled tactile and net-wrench baselines constrain interpretation.
The key is already seated in the pre-turn window; this is not a guaranteed
contact-free control. The inherited feature name `precontact` denotes the
1.05–1.45 s preload reading in this experiment.
Tactile noise/bias/drift/delay/saturation exports are reproducible stress tests,
not calibrated GelSight noise. `sensor_stress_probes.json` separately evaluates
them on the same held-out grip groups, with dense-field and net-wrench inputs
derived from the same corrupted readings. The ideal-sensor head comparison is
kept separate. Small pilot probe results are exploratory.

## Run and review

The reviewed gallery is [output/benchmark_representatives](output/benchmark_representatives/README.md):
18 isolated variants and one insert–turn composite. To package the same captured
movies into a new folder (validates eligibility, duration, full decoding and hashes):

```bash
.venv/bin/python experiments/tool_use_pilot/benchmark_gallery.py \
  experiments/tool_use_pilot/output/task_verification_20260909 \
  --output /path/to/new_gallery
```

`representative_episodes.json` records the selection. Repair-specific one-off
recipes have been retired; use `--specs` for explicit new bounded sweeps.

From the repository root, use a new output folder or unique phase name:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  experiments/tool_use_pilot/benchmark.py --output /path/to/pilot \
  --phase screen --workers 8

# Other recipes: qualification, canonical, diversity, randomized,
# sensing and (after the gate passes) composite. --specs accepts a JSON list.
.venv/bin/python experiments/tool_use_pilot/benchmark_report.py /path/to/pilot
.venv/bin/python experiments/tool_use_pilot/benchmark_branches.py /path/to/pilot
.venv_probe/bin/python experiments/tool_use_pilot/benchmark_probes.py \
  /path/to/pilot --window after
.venv/bin/python experiments/tool_use_pilot/benchmark_report.py /path/to/pilot
```

The persistent locked ledger caps the **entire output root** at 256 attempted
trajectories, including failures. Recipes are alternatives, not a command to run
all recipes indiscriminately. Randomized screening uses four conditions per
family. No phase is overwritten. Video replays do not add new conditions.

```bash
# Independent physics + RGB/tactile replay; checks arrays against the source.
.venv/bin/python experiments/tool_use_pilot/benchmark.py \
  --output /path/to/pilot/physics_media --replay /path/to/episode.h5

# Faster RGB rendering of recorded rigid poses (explicitly labeled).
.venv/bin/python experiments/tool_use_pilot/benchmark_media.py \
  /path/to/episode.h5 --output /path/to/pilot/media
```

The fast renderer does not reconstruct gel interior deformation and is not an
independent physics replay. Numerical replay checks do not replace visual
inspection for orientation, collision representation or camera mistakes.

Current results live in [output/diversity_pilot](output/diversity_pilot):
`REPORT.md`, `report.json`, `quality.png`, per-phase manifests/source snapshots,
HDF5 and outcome JSON, and representative media. During collection the report
is explicitly a snapshot; unresolved ledger IDs show remaining work. Concurrent
phase wall times are not an isolated throughput benchmark.

## Project relevance

Keep variants that produce stable, distinct contact transitions and a meaningful
hidden-state question. Repair grasp or geometry errors before probing sensing.
Physically valid task failures are useful ACWM negatives; they need not be
imitation-eligible. Physically invalid trajectories are diagnostic evidence,
not training examples of valid mechanics.
Defer variants whose only distinction is a visible gross miss, or whose material
physics is absent. The next ACWM step is a small frozen train/test split with
held-out geometry and observation conditions, then cross-domain evaluation on
the main server—not indiscriminate expansion of task count here.
