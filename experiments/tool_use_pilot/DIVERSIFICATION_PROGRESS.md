# Geometry diversification and tracking-margin repairs — 10 September 2026

**11 September completion update:** all 36 geometry/pose trials have finished;
35 pass task/physics checks (friction turning is now 6/6). The old hook roll remains
rejected. Curved-slot, pushing and the new leaned-wall/rail-free-box outcomes are
summarized in the [current handoff](../../SUPERDEX_BENCHMARK_HANDOFF.md).
The checkpoint below is historical; its pending counts and derived review ledger
must not be used as the final release manifest.

## Checkpoint — 16:52 Singapore time (not final qualification)

Implementation and bounded collection are running. **29 of 30 completed geometry/
pose trials pass; six friction-turning trials remain running.** All 30 completed
movies decode fully and were inspected at start/contact/peak/end; all sampled
tool/environment, actual housing and robot-link clearance checks pass. The hook
roll trial is nevertheless rejected for sled tipping and insufficient pull.

| Variant | Completed / planned | Task + physical passes | Force RMSE range | Torque RMSE range |
|---|---:|---:|---:|---:|
| Flat scrape | 6 / 6 | 6 | 0.124–0.130 N | 0.00326–0.00335 Nm |
| Cylinder peel | 6 / 6 | 6 | 0.0388–0.0408 N | 0.00206–0.00228 Nm |
| Friction hook | 6 / 6 | 5 | 0.160–0.169 N | 0.00440–0.00474 Nm |
| Round insertion | 6 / 6 | 6 | 0.162–0.168 N | 0.00369–0.00420 Nm |
| Key insertion | 6 / 6 | 6 | 0.160–0.171 N | 0.00371–0.00428 Nm |
| Friction turning | 0 / 6 | pending | pending | pending |

Wrench ranges include completed attempted trials (including failed hook roll),
and remain force-field diagnostic comparisons, not acceptance gates or a sensor
calibration claim. Hook geometry alone passes 4/4, but the six-condition hook
envelope is not approved. Four complete envelopes are currently approved:
scrape, peel, round insertion and key insertion. No combined collection starts.

Repaired straight guides and preferred pose-push path D are saved in
`output/diversification_v21/selected_repairs_1.manifest.json`. This is an explicit
reusable preset for `benchmark.py --specs`, **not a change to historical baseline
recipes or an override of randomization holdbacks**. Curved-guide first/follow-up
trials are still running and are not included. No further repair attempts are
scheduled beyond the already launched pair.

Data and captured RGB/tactile movies: [campaign](output/diversification_v21).
Per-episode audit: [report](output/diversification_v21/DIVERSIFICATION_REPORT.md)
and [ledger](output/diversification_v21/review.json). The three listed original
baseline serialization errors are historical and superseded, not new failures.
There are no new setup/serialization errors among the completed current trials.

Representative repaired videos:
[pose pushing D](output/diversification_v21/push_centered/0065_pushing_pose_path_d_rgb_tactile.mp4),
[straight wall](output/diversification_v21/tracking_repairs/0058_surface_straight_wall_nominal_rgb_tactile.mp4),
[straight slot](output/diversification_v21/tracking_repairs/0059_surface_straight_slot_nominal_rgb_tactile.mp4).
RGB review sheets are in the campaign's `geometry_media_*` and `repair_media_1`
folders. Original source snapshots, HDF5 and unsuccessful attempts are preserved.

Code verification: 103 tool-use unit tests (101 passed, two skipped), plus 13 gel
tests passed; Ruff F/E9 checks and `git diff --check` pass. No commit or cleanup
deletion was requested or performed in this follow-up.

## Scope

The reviewed revision-21 baseline has 19 successful completed variants, with
three original serialization errors superseded by successful reruns. Videos
were decoded fully and start/contact/peak/end frames reviewed; sampled tool,
fixture and actual housing checks passed. This is nominal feasibility, not
randomized or real-world qualification. No task goal or wrench gate is relaxed.

Proceed only with the six cleaner representatives: flat scraping, cylindrical
peeling, jaw-aligned hook, round insertion, key insertion and friction turning.
`geometry_clean` has 36 candidates: ±10% handle/working size, +5° roll, −5° pitch.
The fixed nominal physics, curved gel and force-only observations are unchanged.

These pose changes preposition the actual wrist and rebuild the prepared grasp
and fixture frame coherently; they are not independent peg/socket alignment
errors. Handle/working size changes are one-factor-at-a-time. Peeler working size
means cylinder radius; insertion/turning scale changes the mating profiles while
preserving clearance. Independent grasp error, fixture misalignment and mixed
physics remain later stages.

All four wall/slot variants and pose pushing are explicitly held out of the
generic geometry/physics/alignment/combined execution driver while repairs are
reviewed. Exported manifests can still describe these candidates; exports are
not approval to collect them. The first clean run uses 12 CPU workers.

## Diagnosed issues and repairs under test

### Guides

At the end of the old curved-slot stroke the wrist target is 20.0 mm, tool root
about 20.6 mm, but actual working-tip motion reaches about 26.1 mm. Loaded grasp
rotation adds roughly 5–6 mm that a wrist/root-position command misses. The curve
is evaluated at the commanded position, so it no longer matches the contact
tip's location. Side contact falls below threshold near 17–18.9 s; overall
tabletop contact remains present. Nominal side-contact fractions were only
84.1%/84.9% for curved slot/rounded wall.

The new guide mode reuses the explicit `tool_pose_feedback` opt-in:

- Follow the known working tip in the along-stroke direction, not just the root.
- Bound integral correction to ±8 mm and its rate to 2 mm/s.
- Evaluate curve position at measured tip abscissa in the fixture frame.
- Keep side/floor-force admittance and Cartesian impedance; do not add a lateral
  position servo that fights contact preload. Correction state is checkpointed.

This uses **oracle rigid tool pose in the collection controller**. The HDF5
metadata explicitly records this; corrected commands and scripted corrections
are retained. It is not a learned tactile controller or a hardware-ready pose
estimator, and tool pose is not added to model inputs. Matching this controller
in real hardware requires a tip-state estimate or another contact-following
controller. The goal here is controlled, useful interaction trajectories.

### Pose pushing

The old 40 mm / 15° goal ends at roughly (34.2, +7.9) mm COM displacement with 17.7° yaw:
9.79 mm position error, just inside the 10 mm acceptance tolerance. Straight
tool travel does not compensate the sideways displacement associated with
rotation. Gentle open-loop curved paths were compared:

| Candidate | Initial lateral contact | Added lateral travel | Forward overtravel |
|---|---:|---:|---:|
| path_a | −19 mm | −6 mm | 11 mm |
| path_b | −18 mm | −10 mm | 12 mm |
| path_c | −10 mm | −10 mm | 11 mm |
| path_d | −12 mm | −10 mm | 11 mm |

The first pair reduced position error but over-rotated the object. The second
pair moved the contact nearer the centre to reduce the moment arm. Both pass
the stronger margin target; **path_d is the preferred nominal candidate**.
Errors use the object's centre of mass, not its mesh/body-root origin.

| Pose pushing | Position error | Absolute yaw error | Goal / stronger margin |
|---|---:|---:|---|
| Old nominal #21 | 9.79 mm | 2.74° | pass / fail |
| path_a #62 | 6.44 mm | 7.64° | fail / fail |
| path_b #63 | 5.23 mm | 10.36° | fail / fail |
| path_c #64 | 3.87 mm | 2.23° | pass / pass |
| path_d #65 | 3.15 mm | 1.86° | pass / pass |

All four candidates are physically valid, full-duration trials with grasp
retention and no tipping or support escape. A/B remain explicitly unsuccessful
task demonstrations; their valid failure data are preserved. C/D captured
movies decode fully, start/contact/peak/end RGB and sampled robot/housing checks
pass. Their largest sampled tool/object/support overlap is 0.103 mm, and grasp
drift is at most 1.33 mm / 0.57°. C/D are two nearby contact offsets at one fixed
geometry/physics, **not broad pushing robustness**.

These modify only the tool path. The object stays a free body; no object-pose
servo, planar constraint, spring or weld is added. Goal stays 40 mm / 15° and
acceptance stays 10 mm / 5°. Precontact alignment is below 21 mm/s, pushing
below 8 mm/s. The new path parameters have historical-safe defaults so old
recordings remain interpretable.

## Verification boundary

`tracking_repairs` contains four nominal guide trials and the first two pushing
candidates, using six CPU workers independently of geometry diversification.
`push_centered` contains C/D and completed in 577.87 s with two workers. All use
captured RGB/tactile recording. No pending repair is assumed successful.

The straight-wall/slot repairs (#58/#59) are complete, task/physics/visual checks
pass. Actual working-tip travel is **20.280 / 20.325 mm**, versus the old
24.019 / 24.023 mm; guide-contact continuity is **100%** in both. Sampled maximum
overlap is below 0.098 mm, with no housing/robot containment hits. Along-stroke
correction reaches about −6.95 mm during following, without hitting its ±8 mm
bound; straight-wall release briefly reaches that bound. Loaded grasp drift
still reaches 6.74 mm / 8.82°: this repairs tracking, not all grasp compliance.
Rounded wall #60 completes with 20.889 mm travel but only **90.62% guide contact**,
below the 95% repair target (old nominal 84.90%). Its late loss coincides with the
lateral correction saturating at 10 mm around t=17 s; guide force falls from
0.15 N to 0.04 N while floor contact remains present. Along-stroke correction
also reaches its bound late. This is improved but not a qualified repair.

`guide_lateral14` is a final bounded pair for rounded wall/curved slot (#66/#67),
allowing 14 mm lateral correction with unchanged 1 mm/s rate, 0.5 N force
targets, ±8 mm along-stroke bound, impedance, grip and acceptance criteria.
Round 3 of `diversification_repairs.py` reproduces this pair. Completion and
review are pending; no further guide tuning is auto-started.

## Geometry/pose boundaries found

All six scrape, peel and round-insertion trials pass the existing task/physical
criteria. Hook geometry ±10% and −5° pitch pass, but **+5° roll (#38) is rejected**:
the sled tilts 12.54°, above the 10° tipping gate, with 8.23 mm guide drift,
and reaches only 38.69 mm of the required 40 mm pull.
The tool remains held (1.15 mm / 1.81° maximum grasp drift), gel Jacobian stays
above 0.884 and no numerical abort occurs. RGB confirms sled tipping rather
than a tool drop. This rejects that support/task outcome, not evidence of a
broken tactile solver. Do not expand the complete hook pose envelope until
the roll/rail interaction is repaired and reviewed; retain the failed attempt.

For stronger margins, aim for guide travel 20±2 mm with at least 95% side contact,
and pose-push endpoint error ≤5 mm / ≤3°. These are additional repair targets,
not retrospective changes to old acceptance flags. Check grasp movement,
penetration, contact continuity, final hold, actual geometry and video, as well
as diagnostic-only force-field wrench error. Limit unresolved repairs to about
one hour per issue before reporting remaining failure.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python experiments/tool_use_pilot/diversification.py geometry --output experiments/tool_use_pilot/output/diversification_v21 --phase geometry_clean --variants flat_scrape cylindrical_peel friction_normal round key friction --review experiments/tool_use_pilot/output/diversification_v21/review.json --run --workers 12
```

That run directory already exists; use a new phase for an authorized rerun. The
repair manifest is generated by `diversification_repairs.py`; it records concrete
parameters and is executed by `benchmark.py --specs`, not the randomized runner.
Neither group automatically starts broader physics collection or main-model
training. Useful next evidence is variation in contact transitions and outcomes,
not merely stronger tactile amplitudes or perfect oracle wrench matching.

## Review tooling

`diversification_review.py --prepare-media NEW_DIRECTORY --media-ids ID ...`
decodes each completed original video fully and creates start/contact/peak/end
RGB sheets. It does not mark visual approval or overwrite the audit ledger.
Use `--visual-reviewed` separately, only after inspection. Housing clearance now
also checks recorded robot-link vertices against tool/fixture meshes; this is a
sampled one-way containment check, not continuous collision certification.
