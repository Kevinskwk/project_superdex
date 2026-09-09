# Bounded task verification — 9 September 2026

Latest follow-up: convex scraping is retired and curved-slot #45 now has a full
pass. The active catalog is **18/18 isolated variants with full nominal
feasibility evidence**, plus the composite. The original bounded sweep below
is retained as historical evidence; see the completion follow-up at the end.

Campaign: [`output/task_verification_20260909`](output/task_verification_20260909).
All attempts, including failed and runtime-censored recordings, are retained.
The generated [report](output/task_verification_20260909/REPORT.md) and
[dataset index](output/task_verification_20260909/dataset_index.json) distinguish
mechanics validity, task success and training eligibility. This is nominal
feasibility verification, not a validated randomization envelope.

Final result within the approximately one-hour repair window: 45 retained
attempts, 11 runtime-censored recordings, no setup errors or unresolved workers.
Seventeen of the nineteen isolated nominal variants have accepted full passes;
the composite passes at both tested spring settings. The remaining variants
are convex scraping (contact-model review) and curved-slot following
(completion-only runtime limitation). No task has been silently dropped.

Validation: 69 regression tests pass; selected lint and whitespace checks pass.
All 45 HDF5 files (119,564 recorded frames, including censored attempts) have
finite, correctly shaped 7×9×3 left/right force fields, 2 mm marker-pitch metadata,
and matching finite action, pose and contact-wrench arrays. All campaign workers
finished; the ledger has zero unresolved reservations.

## Repairs and evidence

| Task / repair | Matched evidence | Interpretation |
|---|---|---|
| Cylinder peeling | #36/#39: 21.35/21.37 mm / 25 mm stroke; blade-first; 100% blade continuity; 0% holder contact; complete release | SDF and mesh both pass at 1 N; sampled tool/cylinder overlap 0.112/0.112 mm |
| Straight wall | #4: 58% guide continuity → #22: 100%, 20.47 mm travel | Heading 90° → 45° balances floor/stroke/side-guide support; same 0.5 N floor load |
| Straight slot | #6: 58% guide continuity → #30: 100%, 20.49 mm travel | Same heading repair confirms across wall/slot geometry |
| Rounded wall | #40: 23.59 mm / 20 mm stroke, 100% guide contact, complete release | Pass at 45° / 0.5 N; travel overshoot should be included in future robustness checks |
| Curved slot | #41: 23.59 mm / 20 mm, 100% guide contact, retained; recorded through 22.72 s in release | Censored before its configured 30.67 s endpoint; not accepted as a full rollout |
| Normal / mirrored hook | #42/#43: 42.75/42.74 mm slider travel against 40 mm goal, both full 32 s recordings | Retained through pull/unload/lower/final hold; peak relative rotation about 12.3° |
| Concave drawing | #3: 27.32 mm → #29: 18.47 mm, for a 20 mm stroke; 100% contact | Working-tip XY rotation compensation removes root-pivot overshoot |
| Convex scraping | #27/#44: 17.11 mm at 0.5 N with 4/16 iterations; #28/#37: 14.80 mm at 1 N | Lighter load improves motion, but contact-manifold review remains unresolved after the solver test |
| Five insertion profiles | #10–14: round, square, hex, D, key all complete and pass | Current gel/controller feasibility, one trial per profile |
| Three turning loads | #15–17: spring/friction/detent all complete at 91.51°/92.24°/92.38° | New turn-and-hold endpoint verified; no return or withdrawal |
| Native friction resistance | #16 → #33: force peak 3.63 → 0.50 N; moment peak 0.299 → 0.0287 Nm | Same 100° command and 0.015 Nm friction setting; native joint model avoids the large externally driven transient |
| Insert–turn–hold | #25/#26: final 91.53°/92.11°, seated at about 16 mm | Both spring settings pass continuous insertion-to-turn handoff; no pose reset |

The curved-slot check #41 used its 1350 s wall budget and remains incomplete.
This is a recording/performance limitation, not a demonstrated drop or contact
failure; do not drop the task solely on this evidence. It needs a completion
rerun with sufficient wall budget, or a prospectively specified shorter idle
tail that still includes the complete release. Earlier #8/#9 and #31/#32 reached their
motion goals but hit wall-time limits; they are **not** accepted full episodes.
The 32 s hook reruns keep the full pull/unload/lower sequence, then about 4.6 s
in the final stationary state; only excess idle time is removed.

## What changed physically

- The peeler's lower holder ends previously met a plane before its recessed
  blade. The new closed, 40 mm-diameter, 120 mm-long cylinder fits inside the
  blade span. Its axis follows the peeling stroke, and its crown is positioned
  from the central blade edge. The actual SCFields aperture is preserved.
- Blade and holder loads are classified from actual contact locations in the
  tool frame. Holder-only contact cannot satisfy the task. These are oracle
  diagnostics, not tactile model inputs.
- Guide headings were swept at 30°/45° and floor loads 0.25/0.5 N. All four
  straight-wall cases pass, but 45°/0.5 N follows the requested stroke more
  closely without reducing the nominal floor load.
- Curved-surface XY compensation keeps rotation about the tool root from adding
  unintended tip travel. Existing minimum-tip-travel and lateral/contact limits
  remain unchanged. The redundant root-travel gate was removed: compensated
  concave following needs only about 7–8 mm root translation for an 18.5 mm tip
  path. Raw historical scores are preserved; derived reports recompute them.
- SDF versus mesh collision was compared with the same saved curved geometry.
  SDF peeling passes; changing the collider alone did not fix convex scraping.
- Revision 18 records the selected defaults. Revisions 16/17 retain historical
  values, and each campaign phase has a source snapshot and full configuration.

## Tactile diagnostics, not gates

Force fields remain two 7×9×3 arrays with 2 mm marker spacing on the curved gel.
The matched flat gel is a reference. Wrench agreement never rejects episodes.
For illustration, full-episode force-only wrench RMSE for native-friction
turning (#33) is 0.1725 N / 0.00277 Nm, compared with 0.2929 N / 0.01441 Nm for
external friction (#16). This supports a numerical/controller improvement,
not independent real-sensor calibration. Wrenches use world axes about the tool
COM with the recorded Newton–Euler compensation.

The 16-iteration native-friction check #35 is censored at 13.59 s, during
turning. Over its shared post-initialization prefix with four-iteration #33,
rotor-angle RMS difference is about 0.097°; left/right field component RMS
differences are 5.82/4.33 mN, with individual transient differences up to
about 0.22 N. This is partial sensitivity evidence, **not full convergence**.
Its incomplete rollout is excluded from accepted data.

The convex 16-iteration run #44 did complete. It did **not** resolve the contact
diagnostic: mean oblique-normal load fraction is 73.6% versus 74.0% in #27, and
the peak per-contact Coulomb-excess diagnostic is 0.390 versus 0.273 N. Sampled
overlap remains about 0.386 mm. Obliqueness relative to fixture Z is not by itself
a violation on a curved surface, but the broad-blade edge/manifold behaviour is
still unexplained. The variant remains excluded pending dedicated contact-model
review, not because its tactile wrench RMSE misses a threshold. It is not proven
infeasible; the bounded repair did not establish trustworthy contact mechanics.

## Visual review and limitations

Actual simulator RGB was inspected for cylinder blade contact, guide following,
concave/convex contact, retained hook geometry, and the seated composite endpoint.
The new cylinder passed full-history sampled tool clearance and a separate
30-frame robot-vertex/cylinder check with no positive vertex overlap. These
sampled checks are not proofs of zero intersection.

Representative [cylinder-peeling RGB + tactile video](output/task_verification_20260909/surface_sdf/0036_surface_cylindrical_peel_nominal_rgb_tactile.mp4).
Its [force-only tactile/contact wrench plot](output/task_verification_20260909/peeling_wrenches.png)
has full-episode RMSE 0.0439 N / 0.00271 Nm; it is a diagnostic, not the pass gate.
The cylinder is rigid and fixed: this is **peeler contact/surface following**,
not skin fracture or material removal. Drawing likewise does not deposit ink.

Prepared grasps, privileged initial calibration and collector force feedback
remain explicit. These runs do not demonstrate a learned tactile policy,
sim-to-real transfer or tactile benefit over noisy vision. Their project value
is establishing physically plausible tasks and clean labels before those
controlled comparisons and large-scale training.

Do not pool attempts into a benchmark success rate: they mix old defaults,
repairs, load sweeps, solver tests and censored runs. Timings include rendering
when enabled and overlapping CPU contention; this is not a throughput benchmark.

## Completion follow-up and flat-only scraping

User decision: retire convex scraping; retain only flat scraping in active
recipes. The generic geometry/replay paths and all historical results are
preserved, but retired convex episodes are excluded from active eligibility.
Concave drawing, cylinder peeling and the other surface-following tasks remain.

Curved-slot episode #45 used the exact #41 configuration except for episode ID,
output block and wall allowance (1350 → 3600 s). The simulation duration stayed
30.67 s; no motion, material, collider or success criterion was changed.

| Completion metric | Result |
|---|---:|
| Recorded duration / frames | 30.67 s / 3067 |
| Wall time, including rendering | 1815.6 s (30.3 min) |
| Following / guide-contact continuity | 100% / 100% |
| Working-tip travel / commanded stroke | 23.59 / 20 mm |
| Lateral-error p95 | 1.852 mm |
| Peak relative grasp translation / rotation | 5.958 mm / 7.216°; retained |
| Independent full-history floor overlap | 0.0794 mm maximum |
| Sampled all-environment overlap, including slot walls | 0.0974 mm maximum |
| Final 0.3 s median net contact force | 0.000466 N |
| Force-only wrench RMSE, diagnostic only | 0.1310 N / 0.00757 Nm |

The previous 2272-frame prefix matches **exactly** in actions, tool poses,
both tactile fields, working-tip positions and contact wrenches. The completed
force fields are finite arrays of shape `(3067, 7, 9, 3)` per finger with 2 mm
marker-pitch metadata. Seventy regression tests and the selected lint checks pass.

Actual RGB and side views were checked at acquisition and late release. The
tool remains held. Light side contact seen at 23 s decays below the contact-event
threshold by the final hold. This release unloads contact; it is not a commanded
full extraction above the slot walls. The 18% stroke overshoot remains relevant
to future path-accuracy/perturbation testing, despite passing the existing task
criteria. No broad robustness or real-sensor calibration claim follows.

[Completed curved-slot RGB + tactile video](output/task_verification_20260909/curved_slot_completion/0045_surface_curved_slot_nominal_rgb_tactile.mp4)
and [HDF5 recording](output/task_verification_20260909/curved_slot_completion/0045_surface_curved_slot_nominal.h5).
