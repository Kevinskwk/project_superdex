# SuperDex contact-rich tool-use benchmark — agent handoff

Updated **11 September 2026, Singapore time**. Audience: agents continuing on the
main server with the AC-VTWM / unified latent WAM code and other simulators.
This is the current handoff, not a claim that the full benchmark or training
dataset is frozen. Paths below are relative to the `project_superdex` repository.

## Read this first

- Embodiment: **FR3/Franka arm + Franka parallel gripper + two soft GelSight Mini
  pads**. The initial Robotiq experiment is historical, not the current robot.
- Keep **FP32**, **source-fitted curved gel**, **2 mm marker pitch**, and **force-only
  7 × 9 × 3 per pad** as defaults. Flat gel is a reference. Do not add cell-moment
  channels to model inputs or reject episodes for imperfect wrench agreement.
- Revision 21 has **19/19 isolated variants with nominal task/physics passes**
  after replacement runs. Its bounded geometry/pose screen is **35/36 passing**;
  the old guided-hook +5° roll tips the sled. Neither number establishes broad
  robustness, independent random-trial success rate, or sim-to-real performance.
- Latest revision-22 repairs: **20° leaned rounded wall passes**; rail-free
  loaded-box pulls pass for **sideways 0.2 kg, sideways 0.4 kg, and top-down**.
  **10° world-roll loaded-box pull remains rejected** for excessive relative
  tool/gripper rotation. It reaches the box goal but fails grasp validity.
- All previously running simulations have finished. No new full rollout or training
  was launched for this handoff/packaging. A clean-checkout initialization/render
  check was used for portability. Latest four hook recordings were rechecked from
  completed HDF5/JSON, fully decoded, visually sampled, and geometry-audited today.
- **Do not start large collection yet.** Freeze the selected configuration set,
  adapter/action timing, observation allowlist, branch semantics and grouped
  splits, then run a small SuperDex–MuJoCo end-to-end training/transfer pilot.
- **Packaging update:** mechanics/diversification and asset/replay reproduction
  are now committed on `feature/gelsight_mini`. Use the commits adding
  [REPRODUCING.md](experiments/tool_use_pilot/REPRODUCING.md), not the older pilot
  checkpoint. SCFields meshes and raw evidence remain ignored and do not travel
  with Git; the pinned asset preparer rebuilds the required imported tools.

## 1. Research purpose and decisions

The primary question is whether shared action-conditioned latent dynamics transfer
useful interaction consequences across domains and reduce target-data requirements.
Structured bridge interactions and state-matched alternative actions should be
compared against strong target-only and ordinary heterogeneous-cotraining controls.
The unified WAM shares policy and transition computation; task simulation is an
enabler, not the central contribution.

Use the suite to cover sustained normal/shear contact, geometric constraint,
engagement/tension, insertion/seating/torsion, and visible object dynamics. Do not
invent many semantic names for essentially the same scraping motion. Tactile is
optional information: it may resolve hidden load, seating, slip or recovery
ambiguity, but **not every task must require tactile**. Pushing is a useful
visual-dominant comparison, not yet a proven vision-only transfer result.

User decisions to preserve:

- Prefer setups reproducible with printed rigid tools/fixtures and simple friction
  loads. Keep spring/detent variants as optional simulator experiments.
- Scraping should be supported across the fingers rather than pull the tool out
  along its shaft. Use gentle motion and impedance, not a welded tool.
- Peeling means blade-first following on a convex cylinder resembling produce;
  it does **not** simulate cutting or material removal. Flat peeling and convex
  scraping are retired. Flat scraping remains active.
- Key tasks end **turned and held**; no return/unturn is required. Keep insertion
  and turning separately testable; composite insert–turn–hold is an extension.
- Thicker shafts/tips improve real RGB-D observability. Independent handle/tip
  shape/size and gripper-pose diversity are desirable, but only tested ranges
  are currently supported.
- Closed-loop mirrored hook duplicates are removed. Rail-free loaded-box pulling
  is the latest real-matching replacement candidate.
- Levering/spatula/pancake serving is **deferred**, not part of the active roster.
  A future pancake needs a convex underside and opposing support wall; do not
  restart the old flat-edge scooping repair by default.
- Stop unproductive task tuning after roughly an hour per unresolved issue and
  report the limitation. Prefer contract/transfer evidence over perfect force plots.

Project context was reread in Notion for this handoff; see section 11. This session
does not verify the state of the main server's model implementation or datasets.

## 2. Repository and runtime handoff

Source workstation: `/home/showlab/project_superdex`, branch
`feature/gelsight_mini`. The pre-packaging HEAD was
`0ba18f02853f5a1ba855188288bf047caf61de23`; use the newer commits below.

Latest committed checkpoints:

| Commit | Meaning |
|---|---|
| `ea7f776` | Source-fitted curved gels and 2 mm force grids |
| `753fefa` | Verified historical pilot and representative video gallery |
| `0ba18f0` | Pruned superseded pilot artifacts with recovery audit |
| `2dced20` | Real-transfer mechanics, loaded-box/guide repairs and gated diversification |
| `061fa50` | Pinned portable assets, frozen recorded replay specs and expected results |

**The revision-19–22 follow-up implementation is now committed**, including task
worlds, controllers, diversification, audits and tests. This supersedes the
uncommitted-worktree warning in the earlier Notion attachment. Run
`git status --short` on both machines before reset/cleanup; do not discard new
user changes. The packaging request created local commits, not a remote push.
No existing experiment output was deleted.

Local working runtime is Python **3.12.11**, wheel packages
`superdex==superdex-physics==superdex-robotics==superdex-studio==1.0.0`.
Selected installed dependencies: NumPy 2.5.2, SciPy 1.18.1, h5py 3.16.0,
trimesh 5.0.0, matplotlib 3.11.1, imageio-ffmpeg 0.6.0, rtree 1.4.1.
Recreate a Python 3.12 environment on the server; do not copy `.venv` blindly.
Use the repository's wheel-install instructions, **not an incidental source
build**. `uv run --no-project` avoids root-project source resolution. `uv` was
not on this shell's PATH at handoff; `.venv/bin/python` works directly.
The desktop launcher is `.venv/bin/superdex-studio` and requires a display.

Physics/FEM runs on **CPU**, independently in worker processes; captured RGB uses
the offscreen EGL GPU renderer. `.venv` contains CPU Torch; GPU Point-M2AE probes
used separate `.venv_probe` and
[requirements-decision-probes.txt](experiments/tool_use_pilot/requirements-decision-probes.txt).
Do not replace simulation dependencies just to enable GPU training.

Portability checklist:

- Transfer the latest committed branch. Recorded replay configurations and
  expected results are now in `experiments/tool_use_pilot/reproduction/`; a final
  training release manifest/split protocol still needs qualification.
- Copy/recreate ignored `assets/scfields/` separately; **never commit SCFields
  assets**. The preparer is
  [scfields_assets.py](experiments/gelsight_mini_contact_validation/scfields_assets.py).
  Its immutable-revision lock reproduces all 27 imported tools and checks source,
  collision and surface hashes; manifests are relocatable. No TacSL checkout or
  capsule assets are needed for the current pilot. HydroShear source/generated
  sensor assets are already in the repo. See `REPRODUCING.md` for setup.
- Preserve chosen HDF5, completion JSON, source snapshots, manifests, metrics and
  videos. Some historical dataset indexes use **relative HDF5 external links**;
  move the index and shards together. Do not assume ignored outputs travel by Git.
- Output sizes here: tool-use pilot approximately **15 GB**, sensor-validation
  output **4.2 GB**, SCFields subset **7.6 MB** (`du -sh`, approximate).
- GPU drivers/EGL and codec availability need a one-episode check on the server.

## 3. Current task inventory and evidence levels

Revision-21 isolated roster (`diversification.baseline()`):

| Family | Variants | Status / boundary |
|---|---|---|
| Surface following | `flat_scrape`, `flat_draw`, `cylindrical_peel`, `concave_draw`, `straight_wall`, `rounded_wall`, `straight_slot`, `curved_slot` | Eight nominal passes; use the stronger guide repairs below, not old wrist-only tracking |
| Hook | `friction_normal` | Old jaw-aligned loose-guided sled nominal passes; +5° roll fails. Prefer new `loaded_box` candidate for real matching |
| Insertion | `round`, `square`, `hex`, `d`, `key` | Five nominal passes; geometry factor checks only cover round/key |
| Turning | `spring`, `friction`, `detent` | Three nominal turn-and-hold passes; friction is simpler for real matching |
| Tool-mediated pushing | `translation`, `pose` | Two nominal passes; pose path D improves endpoint margin |
| Composite | friction/spring insert–turn–hold | Implemented, but current-version prerequisite/repeat/branch qualification is unfinished; historical passes are not current robustness evidence |

### Bounded geometry/pose screen, revision 21

`experiments/tool_use_pilot/output/diversification_v21/geometry_clean/`:

| Representative | Task + physical passes / attempted |
|---|---:|
| Flat scraping | 6 / 6 |
| Cylindrical peeling | 6 / 6 |
| Old friction hook | 5 / 6 |
| Round insertion | 6 / 6 |
| Key insertion | 6 / 6 |
| Friction turning | 6 / 6 |
| **Total** | **35 / 36** |

Factors are handle size ±10%, working size ±10%, wrist roll +5°, pitch −5°,
one at a time. These are deterministic bounded conditions, not a random success
rate. Legacy pose changes rotate the prepared fixture frame together with the
grasp: they are **not independent peg/socket alignment errors**. Old hook #38
tilts the sled 12.54° (>10° limit), reaches 38.69 mm (<40 mm), but retains the tool.
Turning #52–57 final metrics all pass; the campaign's older review ledger still
needs synchronization before automated stage promotion.

### Latest completed repairs, revision 22 / targeted revision 21

All episode IDs in this table belong to `output/diversification_v21/`.

| Case / ID | Measured outcome | Qualification |
|---|---|---|
| Straight wall #58 | 20.280 mm tip travel; 100% guide contact | Stronger nominal margin pass |
| Straight slot #59 | 20.325 mm; 100% guide contact | Stronger nominal margin pass |
| Curved slot #61 | About 20.15 mm; 100% guide contact | Stronger nominal margin pass; #67 duplicate is not an independent repeat |
| Rounded wall, upright #66 | 22.186 mm; 100% contact; 3.064 mm lateral p95 | Fails stronger tracking margin |
| Rounded wall, 10° lean #78 | 20.081 mm; 82.83% contact; 0.703 mm lateral p95 | Fails ≥95% guide-contact target |
| **Rounded wall, 20° lean #79** | **21.176 mm; 100% contact; 0.229 mm lateral p95** | **Preferred nominal repair** |
| Pose pushing, path D #65 | 3.15 mm position / 1.86° yaw error | Stronger ≤5 mm / ≤3° margin pass |
| **Loaded box, sideways, 0.2 kg #84** | **50.45 mm pull; 6.11° relative tool/gripper rotation** | **Pass; preferred hook orientation** |
| **Loaded box, sideways, 0.4 kg #85** | **49.52 mm pull; 6.30° relative rotation** | **Pass; heavier-load check** |
| Loaded box, top-down #86 | 47.06 mm pull; 13.65° relative rotation | Pass, but limited margin to 15° bound |
| Loaded box, 10° world roll #87 | 43.48 mm pull; **16.34°** relative rotation | **Rejected** despite reaching goal; do not relax the gate |

Latest #84–87 all finish **3,067 steps / 30.67 s** with no numerical/safety abort,
no box tipping or support escape. Maximum box tilt is <0.006°. Maximum sampled
mesh overlap is respectively **0.066 / 0.062 / 0.059 / 0.058 mm**; actual housing
and robot-link containment is zero at the 11 audited frames. Full videos decode;
start/contact/peak/end RGB was viewed on 11 Sep. These are sampled checks, not
continuous collision certificates. #87 remains excluded under existing physical/
grasp-validity labels; relative rotation alone is not an interfacial slip detector
and the images do not show the tool falling out.

For #84/#85, median horizontal resistance is **0.592 / 1.062 N**; p95 absolute
vertical force on the box is **0.365 / 0.552 N**. The heavier box increases
resistance as intended. Two loads are a useful mechanics check, not a calibration
curve or proof of a learning benefit.

Latest force-field-plus-dynamics versus direct contact **vector RMSE**:

| Episode | Force (N) | Torque (Nm) |
|---|---:|---:|
| Wall 20° #79 | 0.1238 | 0.005590 |
| Sideways 0.2 kg #84 | 0.1714 | 0.003075 |
| Sideways 0.4 kg #85 | 0.1770 | 0.003191 |
| Top-down #86 | 0.1547 | 0.002652 |
| Rolled #87, rejected grasp | 0.1568 | 0.002636 |

These use `dynamic_inferred_extrinsic_wrench` (compact force bins, **no cell-moment
correction**) against `extrinsic_contact_wrench`, excluding initialization. They
include free/contact phases, are simulator-internal diagnostics, and do not
override physical validity. Read phase metrics for contact-only interpretation.

### New loaded-box mechanics

One free rigid tray, **120 × 100 mm**, 6 mm base, retaining walls and integral
loop with fixed **55 mm crossbar height**; mass 0.2/0.4 kg represents fixed ballast.
There are no rails, planar constraints, springs, object-pose servos or anti-tip
forces. A straight shank with J end starts **inside the loop with clearance**;
outside-loop search/entry remains unimplemented. Commanded horizontal pull is
54 mm; success is ≥40 mm held. Table axes stay level.

Sideways is −90° pitch **about task Y**, not world Y. Earlier world-axis sideways
prototypes #76/#77/#82/#83 had initial tool/loop or housing interference and lost
grasps; exclude them. #68–73 were interrupted setup diagnostics, not data.
Current preflight rejects sampled initial overlaps above 0.1 mm.

All new box pulls use collector-only vertical force relief (≤0.5 mm/s, ±6 mm).
Top-down/roll #86/#87 additionally use oracle working-end X tracking (≤1 mm/s,
±12 mm) to compensate grasp deflection. Sideways #84/#85 do not use this dynamic
tip tracking. One-time settled grasp-translation calibration remains privileged.
No gel hardness/friction/grip increase was used for these repairs.

The 20° wall uses actual wrist lean, preserved working-rim geometry, tip-based
curve abscissa and bounded tip/force feedback. It still reaches 8 mm tangential
and 10 mm lateral correction bounds and ~11.12° grasp deflection: nominal pass,
not a broad pose envelope. Independently level fixtures currently apply only
to guide surfaces and loaded boxes. Fixed physical table Z plus arbitrary wrist
poses still requires position/IK sampling; fixture height currently follows the
prepared tool height.

## 4. Gel, sensing and controller semantics

- HydroShear sensor/finger/gel meshes are from commit
  `f815b82fdf3451852acd918933020a82cede1f3b`, with MIT license and checksums in
  [THIRD_PARTY.md](assets/bots/grippers/franka_gelsight_mini/THIRD_PARTY.md).
  Soft-skinned tetrahedral attachment follows the original SuperDex soft-finger
  approach. The narrowing exposed side faces the grasped object; backing attaches
  to the housing. The source-fitted surface is curved, not a uniform flat box.
- `GelMaterial(geometry="source_surface")`: **297 FEM nodes, 960 tetrahedra**;
  nominal E=200 kPa, Poisson ratio 0.49, density 1000 kg/m³, mass damping 5/s,
  stiffness damping 0.003 s, effective gel/tool friction 1.4. These are synthetic
  nominal parameters, **not calibrated latest-standard-GelSight material data**.
- `matched_box` is the same-XY-topology flat reference. `legacy_box` is the old
  bounding-box model with ~3.46 × 3.16 mm grid spacing; do not label it a 2 mm gel.
- Marker lattice: 7 × 9, **2 mm projected XY pitch**, 12 × 16 mm span; curved 3D
  neighbor distance can be ~2.002 mm. Force bins sum mapped exposed FEM contact
  loads into nearest-marker cells. They are not optical marker-to-force inference.
- `SoftGripper.get_dense_contact_field(side)` returns all-node current world
  positions and forces, including shoulders; register `NODE_POSITIONS` and
  `NODE_CONTACT_FORCES` before stepping. `get_surface_force_field(side)` returns
  `(7,9,3)` **forces ON GEL in its sensor frame**, in N (not pressure).
- Binning preserves summed mapped force but not exact moment arms. `sum(r×F)`
  from compact bins can estimate torque, but cannot reproduce every dense-field
  moment. Cell moments/unmapped loads are diagnostics only, not model features.
  Dense nodal arrays are optional (`record_dense_field`); their integrated wrench
  diagnostics can exist even when the full arrays are not recorded.
- Wrenches compare forces **ON TOOL**, world axes about instantaneous tool COM.
  Rotate each pad's forces and negate gel reactions before summing. Dynamic
  extrinsic inference subtracts the gel-on-tool wrench from
  `[m(a−g), Iα + ω×Iω]`. A wrench about a different origin needs the corresponding
  `r×F` shift. Finite-difference dynamics and marker binning add error.
- CoP is a force-weighted contact-location / wrench-derived line-of-action
  diagnostic; it is not a unique contact point for arbitrary distributed shear
  and free torque. It is ill-conditioned at small normal force.
- Curved geometry is retained for geometric realism, **not because matching real
  sensor error proves fidelity**. Simulator agreement is internal consistency,
  not independent validation of a real tactile sensor.
- Coupled Franka fingers, ≤35 N **per finger**; actual arm Cartesian impedance
  K/D translation 1100 N/m / 85 Ns/m and rotation 40 Nm/rad / 4 Nms/rad for current
  candidates. Arm/gripper gravity and motor friction are ideally compensated;
  tool gravity remains enabled. This is not a calibrated hardware controller.
- Housing/tool collisions are intentionally disabled; gel/tool contact is active.
  Separate mesh audits compensate partly for this known modeling limitation.
- Surface loads/guide following and some grasp/tip corrections use simulator
  oracle feedback for collection. They are not deployable tactile-only policies.

## 5. HDF5 / model adapter: critical integration contract

Current benchmark schema: **`vt_acwm_superdex_benchmark_v1`**, one `.h5` plus
completion `.json` per episode. Older `vt_acwm_superdex_tool_pilot_v1` and parallel
collectors have different storage/semantics; never dispatch solely by file suffix.

| Channel | Current meaning / adapter rule |
|---|---|
| `geometry/<body>/{vertices,faces,root_to_com}` + `observations/body_root_poses` | Saved mesh and per-frame body-root pose; role metadata separates tool/environment/occluders |
| Point clouds | **Not stored per frame in current benchmark HDF5**; sample meshes and transform with body-root poses, then apply visibility/corruption offline |
| `tactile_force_field_left/right`, aliases `tactile_data_*` | T × 7 × 9 × 3, sensor-frame N; model force-field input |
| `tactile_coord_*`, `tactile_displacement_*` | World-metre marker positions / sensor-frame displacement; do not silently add oracle deformed geometry to a force-only model |
| `ee_pose`, `tool_pose`, velocities | SI world-frame, XYZW quaternion; tool pose is COM, not mesh-root or working-tip pose |
| `actions`, `ee_target_pose`, efforts | Absolute EE translation offset in initial task/tool axes + rotation-vector offset in radians; executed command, **not per-step Cartesian delta**, torque action or joint command |
| `timestamps`, `phase`, reference/reset flags | 100 Hz raw samples in current trials; initialize/exclude grasp-settle frames and preserve causal reference history |
| Wrenches, fixture state, progress, slip, solver/controller channels | Evaluation/collector oracles; exclude from ordinary vision/tactile model inputs unless an explicitly named oracle ablation |
| `labels`, completion metrics | Physical validity, grasp retention, tactile diagnostic, task success, eligibility remain separate |

**One-step timing hazard:** `step(t)` computes/applies command, advances physics,
then records the observation at `t+dt` in the same row as that command. Thus stored
row `i` is `(a_i, o_{i+1})`. For consecutive recorded observations, pair
`(obs[i], actions[i+1], obs[i+1])`. Preserve the first pre-step observation from a
checkpoint or explicitly discard the unavailable initial transition. Verify
chunk/downsampling alignment with an impulse/hold test in both simulators;
matching six-column shapes is not sufficient.

Read `action_semantics`, `action_frame_rotation_world`, initial EE pose and
`config_json`. The stored command includes collector correction, so retain a
distinction between nominal program and executed command. H1/H5 in historical
probes meant 100/500 ms at 10 Hz, not 1/5 frames at 100 Hz; define horizons in time.

**Revision-22 frame hazard:** saved tool geometry bakes in prepared wrist tilt.
For rendering/PCD, apply the saved mesh root pose **once**. For authored tool axes:
`R_world_authored = R_world_mesh @ R_task_initial.T @ R_grasp_initial`.
The constant is in newer `mesh_from_authored_tool_rotation`; earlier #84–87
snapshots can derive it from `action_frame_rotation_world` and
`prepared_grasp_rotation_world`. Body-root versus COM offsets also matter.

Observation pipeline: ray-visible sampled geometry with reproducible XYZ noise,
persistent registration bias and camera-plane patch removal. Current preview
levels are 1/2 mm noise and 15/30% initial visible-cloud removal. Masks are fixed
after initial calibration, not selected from contact labels. Fully occluded
streams can occur and are reported. This is **not a calibrated RGB-D sensor model**.
Exact mesh/pose is the rendering backend, not permission to feed GT pose to a
vision baseline. Recorded RGB is actual EGL simulation RGB; review videos do not
imply optical GelSight RGB generation or learned force reconstruction.

Split by underlying geometry/physics/configuration and branch ancestry. Keep all
observation augmentations and sibling branches together. `config_json` includes
geometry/physics/split identities; legacy finalized datasets may instead expose
`split_group_id`. Hash **actual meshes** to catch duplicates across labels.
The existing 0.95 geometry / 1.1 load holdout definitions are conditional factor
holdouts, not universally geometry-disjoint partitions. Use train-only statistics
and keep existing locked test sets on the main server locked.

## 6. Counterfactuals, memory and learning evidence

Native physics snapshot/restore plus controller/reference/RNG state and common
history are implemented; historical branch checks demonstrated exact replay.
However **current-release task-specific branch qualification is unfinished**:

- `benchmark_branches.py` still selects old hook `pull/left/right/hold` macros;
  loaded-box semantics now support nominal/hold/retract instead. Unsupported
  names can duplicate nominal motion. Update/verify the mapping before collection.
- `TransferWorld.command_at` pushing currently ignores branch choice. Four branch
  names are not evidence of four different interventions.
- Branch writer metadata is not yet fully synchronized with nominal writer
  (e.g. loaded-box force-feedback labeling). Freeze one common export contract.
- Check exact same-state/same-action replay, identical pre-branch history and
  tactile references, then meaningful different executed actions/effects, sibling
  completeness and safety on each retained family. Test event-based anchors,
  not only arbitrary wall-clock times.
- Physically valid misses/exploration may supervise consequences, **not imitation**.
  Invalid simulation/setup cases remain diagnostic rejects. Under current policy
  #87 is invalid even though it does not drop the tool. If safe slip is later
  desired as training data, define and validate that separate envelope explicitly.

Historical September-7 Point-M2AE screens (earlier task/gel version): hook tactile
helped chiefly under degraded vision. Key hidden-resistance balanced accuracy was
52.1→75.0% clean and 64.6→87.5% with 1 mm noise +25% patch removal. Clean insertion
was already 100% vision-only; severe corruption gave 89.6→100%. Turning torque
MAE was 0.0181 Nm vision, 0.0083 vision+dense tactile, 0.0038 with axis-aware
tactile-derived net wrench. **Only 12 qualified physical conditions per key stage;
no fusion gain survived global Holm correction.** Repeated corruptions are not
independent episodes. This neither proves dense-field necessity nor transfers
automatically to current geometry. See [DECISION_PILOT.md](experiments/tool_use_pilot/DECISION_PILOT.md).

Main-project next evidence: matched vision/proprioception, +force-field, and
tactile-derived net-wrench controls; state-only/zero/shuffled-action controls;
with/without meaningful state-matched branches; target-only, cotrain without
source support, and cotrain with source support at matched model/data/step budgets.
Primary outcomes are target-data efficiency, future physical/event prediction,
action ranking and policy task/recovery performance—not tactile amplitude.

Memory comes after the WAM compatibility gate: current tactile + explicit grasp
reference versus causal window versus recurrent action/tactile history, including
a no-tactile temporal control. Use informative interaction → unload/pause → later
action and audit current-state leakage. Candidate future actions stay outside
the state estimator. Add uncertainty mechanisms only for ambiguity that remains
after sufficient history; no broad uncertainty/planning stack is implemented here.

## 7. What must be done before large collection

1. **Use the committed implementation/replay specs, then freeze the training release.** Historical
   `transfer`, `diversity`, `randomized`, default `BenchmarkSpec` and revision-21
   `baseline()` do not automatically contain latest repairs. `pose_transfer`
   includes the rejected rolled hook as a diagnostic candidate: it is not an
   accepted-only dataset recipe. Root `review.json`, `DIVERSIFICATION_REPORT.md`
   and `selected_repairs_1.manifest.json` predate some final runs. Reconcile the
   ledger from complete recorded evidence; never auto-approve unseen videos.
2. **Resolve scope, not every experimental corner.** Start with flat scrape,
   simple insertion, friction turning, sideways loaded-box pull and pushing.
   Keep #87 roll excluded; use #86 top-down cautiously or defer its wider envelope.
   Retain guide repairs as optional bounded variants until perturbation margins
   are established. Broad independent 6-DOF pose sampling is not done.
3. **Audit the shared data/action/observation contract** using actual main-server
   Isaac/MuJoCo loaders; implement transition alignment, model-input allowlists,
   reference/reset semantics, valid-failure/imitable masks and collision-free splits.
4. **Qualify a small counterfactual package** with repaired branch dispatch and
   replay/history tests. No need for a huge branch bank before checking utility.
5. **Port a few semantic bridges to MuJoCo**, then train a small end-to-end
   comparison. Match geometry, task outcome, action convention, control rate,
   impedance and force-versus-speed response; do not require identical trajectories
   or raw tactile values across solvers. Add real-target episodes once hardware
   observation/calibration is available. Another simulator is optional, not a gate.
6. **Bound physics and geometry sensitivity** on the chosen release candidates:
   effective friction, load, grip, gel modulus, clearance, independent pose error,
   and higher solver accuracy/timestep. Staged manifests exist, but the current
   revision-21 physics/alignment/combined sets are not fully executed/qualified.
   Earlier material sweeps are not blanket qualification of new tasks.
7. **Benchmark throughput on the main server** using the frozen full-length
   tasks and accepted-frame yield, with recording/I/O separately timed; only
   then set production shard sizes, workers, data volume and storage estimates.

Important physics mismatch: Mochi defaults to **10 mm/s friction falloff velocity**
(`friction_falloff_vel`, native `frictionFalloffVel`). Gentle loaded-box pulls are
slower, so measured resistance is below nominal `mu*m*g`. Match a force-vs-speed
curve across simulators; copying coefficient 0.4 alone is insufficient. Effective
pair friction also depends on the engine's material-combination rule. Synthetic
friction sweeps can modify more than one contact pair; read recorded provenance.

## 8. Performance: historical numbers are not current-task forecasts

| Workload on this workstation | Time / rate | Interpretation |
|---|---|---|
| Old plug/table, 128 × 200 steps | 784.92 s end-to-end; 32.61 env-frames/s | Older geometry/collector; includes stored PCD |
| Old cleaned SCFields, 999 × 200 steps | 1064.62 s; 187.67 env-frames/s | Multiprocess/cohort path, different solver/gel/tasks and looser historical angular retention limit |
| Latest two sideways boxes, 2 × 3067 steps, RGB on | 2781.17 s wall (~46.35 min) | Two workers, other jobs overlapped; not isolated throughput benchmark |
| Latest top-down/roll pair, 2 × 3067, RGB on | 2116.11 s (~35.27 min) | One invalid episode; not accepted-data throughput |

Machine: 32 logical CPU cores, ~123 GiB RAM, RTX 5090 32 GB. Recent pilot used
12 independent CPU workers for clean geometry; two-worker repair pairs are not
maximum throughput. Use `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` to avoid
oversubscription, then measure worker/thread affinity on the main CPU. More cores
can help independent scenes, but neither linear scaling nor a Threadripper
speedup factor has been measured. GPU VRAM is for rendering/probes/training, not
FEM state; old whole-device VRAM snapshots include unrelated workloads and are
not collector memory measurements. Do not extrapolate 200-step legacy rates to
30-second curved-gel contact-rich trajectories.

## 9. Code map, evidence and commands

| Entry | Purpose |
|---|---|
| [benchmark.py](experiments/tool_use_pilot/benchmark.py) | Spec validation, task worlds/controllers, collection and annotation |
| [transfer_world.py](experiments/tool_use_pilot/transfer_world.py), [loaded_box.py](experiments/tool_use_pilot/loaded_box.py) | Free-object mechanics and latest rail-free tray setup |
| [benchmark_geometry.py](experiments/tool_use_pilot/benchmark_geometry.py) | Parametric tools, fixtures, guide geometry and task checks |
| [diversification.py](experiments/tool_use_pilot/diversification.py), [diversification_repairs.py](experiments/tool_use_pilot/diversification_repairs.py) | Versioned factor manifests, gates and bounded repair presets |
| [benchmark_report.py](experiments/tool_use_pilot/benchmark_report.py), [diversification_review.py](experiments/tool_use_pilot/diversification_review.py) | Outcome/geometry audits, media review, strong repair margins |
| [benchmark_observations.py](experiments/tool_use_pilot/benchmark_observations.py), [diversification_observations.py](experiments/tool_use_pilot/diversification_observations.py) | Mesh/pose PCD reconstruction, visibility and corruption |
| [benchmark_branches.py](experiments/tool_use_pilot/benchmark_branches.py) | Snapshot branches; current-task dispatch/export must be repaired before use |
| [soft_gripper.py](experiments/gelsight_mini_contact_validation/soft_gripper.py), [tactile_grid.py](experiments/gelsight_mini_contact_validation/tactile_grid.py) | FEM mounting, dense field and compact tactile representation |
| [visuals.py](experiments/tool_use_pilot/visuals.py) | Captured EGL RGB + shear/normal-field video |

Current data root: `experiments/tool_use_pilot/output/diversification_v21/`.
88 recorded/reserved attempt entries (#0–87) span baselines, repairs, deliberate
negatives, interrupted prototypes and errors; **never report their pooled success
rate**. Preserve per-run source snapshots, not just the latest source hash.

Representative evidence (MP4 siblings have `_wrenches.png`, `.h5`, `.json`):

- [Sideways 0.2 kg loaded-box video](experiments/tool_use_pilot/output/diversification_v21/loaded_box_v22_axial/0084_hook_loaded_box_sideways_rgb_tactile.mp4)
- [Sideways 0.4 kg loaded-box video](experiments/tool_use_pilot/output/diversification_v21/loaded_box_v22_axial/0085_hook_loaded_box_sideways_loaded_rgb_tactile.mp4)
- [Top-down passing video](experiments/tool_use_pilot/output/diversification_v21/loaded_box_v22_tip_tracking/0086_hook_loaded_box_top_down_rgb_tactile.mp4)
- [Rolled rejected-grasp video](experiments/tool_use_pilot/output/diversification_v21/loaded_box_v22_tip_tracking/0087_hook_loaded_box_roll_10_rgb_tactile.mp4)
- [20° leaned rounded-wall video](experiments/tool_use_pilot/output/diversification_v21/loaded_box_lean_v22_clearance/0079_surface_rounded_wall_lean_20_rgb_tactile.mp4)
- [Pose pushing D video](experiments/tool_use_pilot/output/diversification_v21/push_centered/0065_pushing_pose_path_d_rgb_tactile.mp4)
- [Latest hook RGB review sheet](experiments/tool_use_pilot/output/diversification_v21/handoff_review_20260911/page_0.png)
- [Historical per-variant gallery](experiments/tool_use_pilot/output/benchmark_representatives/README.md) — revision 18, not a current revision-22 gallery.

Read detailed [loaded-box/wall repair](experiments/tool_use_pilot/LOADED_BOX_AND_LEAN.md),
[diversification design](experiments/tool_use_pilot/DIVERSIFICATION.md),
[geometry/tracking history](experiments/tool_use_pilot/DIVERSIFICATION_PROGRESS.md),
and [revision-18 status](experiments/tool_use_pilot/BENCHMARK_STATUS.md) with their
version labels. Earlier statements of pending runs or all-pass status have local
historical scope; this handoff supersedes them for current continuation.

From repo root, inexpensive regression checks:

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s experiments/tool_use_pilot -p 'test_*.py'
.venv/bin/python -m pytest experiments/gelsight_mini_contact_validation/test_soft_gels.py -q
.venv/bin/ruff check experiments/tool_use_pilot --select F,E9
git diff --check
```

Rechecked after packaging, including a clean checkout on 11 Sep: **113 pilot tests
passed, 2 skipped; 29 asset/gel/collector tests passed**. Ruff F/E9 passed. All
27 imported collision/surface pairs and both curved/flat gel assets regenerated
byte-for-byte. The clean checkout also loaded the relocated cache and initialized
physics, force fields and EGL RGB. Tests are not physical rollout qualification.

Cheap candidate export (no rollout unless `--run`):

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python experiments/tool_use_pilot/diversification.py geometry --output experiments/tool_use_pilot/output/new_server_manifest
```

Bounded latest repair replay, **including the currently failing rolled candidate**:

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python experiments/tool_use_pilot/benchmark.py --output experiments/tool_use_pilot/output/new_server_pose_check --phase verification --recipe pose_transfer --workers 5 --render
```

Do not run this whole recipe merely to test installation; assemble one selected
recorded spec first. Use a new output/phase; do not overwrite historical data.
`diversification_review.py --visual-reviewed` expects the **complete** approved ID
set and can replace prior approvals; media preparation alone does not approve.

## 10. Completion criterion for the next agent

The next useful milestone is a **small reproducible cross-simulator data/training
vertical slice**: selected stable task configurations → correctly aligned and
leakage-safe HDF5 adapter → meaningful shared-state branches → matched SuperDex /
MuJoCo observation-action interface → sanity-trained prediction/policy controls.
Freeze the interface and eligibility protocol before paying for large collection.
Sim-to-real and current-version tactile value remain experiments to conduct, not
results inherited from nominal task feasibility.

## 11. Sources and coordination pages

- [AC-VTWM / Contact World Model project](https://www.notion.so/3a3838adf1828181896efdc39c5bccab)
- [Mixed interaction suite / transfer stability task](https://www.notion.so/3bf838adf1828168973cd714a70c6d7e)
- [Unified WAM + Point-M2AE implementation task](https://www.notion.so/3ab838adf18281c8979fe50f57d11df5)
- [Visual-dominant transfer control](https://www.notion.so/3d1838adf1828139b779ca17d1a0d039)
- [5–18 September sprint](https://www.notion.so/3d1838adf1828160b6cad07021124cda)

Notion updates for this handoff summarize local simulation progress and the next
integration gates; they do not mark the model task, transfer task or sprint done.
Primary numeric sources are the saved episode completion JSON/HDF5 and the local
reports linked above. No main-server files were accessed or changed.
