# Reproducing the current SuperDex pilot on another machine

Use branch `feature/gelsight_mini`, including the commits adding this file.
Read [the handoff](../../SUPERDEX_BENCHMARK_HANDOFF.md) for scientific scope and
known failures. This guide freezes inputs and recorded outcomes; it does not
claim that arbitrary CPUs/solver builds produce bit-identical trajectories.

## Asset strategy

| Asset | Distribution / recreation |
|---|---|
| Franka/FR3 and sensor mounting | Already tracked in Git with upstream provenance |
| HydroShear GelSight source meshes and generated gels | Already tracked, pinned MIT source/checksums; deterministic curved/flat generator below |
| Hook, loaded tray/loop, peg/socket, guide walls/slots, peeler cylinder, push object | Procedurally generated in `benchmark_geometry.py`, `transfer_world.py` and `loaded_box.py` from the saved numeric specs |
| Imported SCFields tools | Download only the locked 27-tool subset from an immutable dataset revision, check source hashes, regenerate collision/visual surfaces and check their hashes |
| HDF5 and captured videos | Remain ignored; regenerate with the committed specs or transfer the original evidence separately |

No new binary asset upload service, Git LFS account or TacSL/IsaacGym checkout is
required for this benchmark. **SCFields meshes remain gitignored**. Do not replace
the peeler with an approximate procedural shape when reproducing these results:
that would change the working edge/holder and contact mechanics.

SCFields lock:
[scfields_assets.lock.json](../gelsight_mini_contact_validation/scfields_assets.lock.json),
dataset revision `923c0b409f65eaefec9ae1ffd5c695aa8aa6d7ca`.
The [upstream dataset card](https://huggingface.co/datasets/Kevinskwk/scfields-release/blob/923c0b409f65eaefec9ae1ffd5c695aa8aa6d7ca/README.md)
declares CC-BY-NC-4.0; keep its attribution/terms separate from the code license.
Sensor asset provenance is in
[THIRD_PARTY.md](../../assets/bots/grippers/franka_gelsight_mini/THIRD_PARTY.md).

The lock fixes the exact tool order and source, collision and visual hashes—not
just a mutable metadata quantile. Current surface representatives are
`tool_scraper_25`, `tool_cylinder_pen_32`, and
`peeler_1_head_rectangular_handle_h0_96_hnd0_99`. The loader names them explicitly.
Changing a checksum or selection requires a deliberate new qualification; never
edit the lock merely to suppress a reproduction error.

## Setup (Linux, Python 3.12, published wheels)

From a checkout containing these commits, create an environment if needed:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r experiments/tool_use_pilot/requirements-simulation.txt
```

If using uv, use `uv pip install --python .venv/bin/python -r ...` and
`uv run --no-project` for wheel-based script execution. Do not accidentally build
the whole native workspace with `uv sync`. Existing environments can be checked
first; do not overwrite a main-model CUDA Torch environment. GPU Point-M2AE uses
its own requirements file/environment and is not needed to collect physics.

Prepare/check assets:

```bash
.venv/bin/python experiments/gelsight_mini_contact_validation/scfields_assets.py
.venv/bin/python experiments/gelsight_mini_contact_validation/scfields_assets.py --check
```

The default creates `assets/scfields/manifest.json` with **relative** paths.
Moving that folder or the checkout is supported. Old absolute manifests are
resolved against their new `scfields` root, never against the old workstation.
Cached files are checksum-checked; corrupt/modified files cause an error and are
not overwritten. Inspect or move conflicting files aside, or use a new
`--output PATH` cache. `--check` performs no network access. Optional
`--capsule-root PATH` imports historical capsule files into a new cache; these
capsules are not required or used by the current task benchmark.

The small HydroShear-derived meshes are already checked in. To deliberately
regenerate the curved and same-topology flat gel:

```bash
.venv/bin/python assets/bots/grippers/franka_gelsight_mini/generate_surface_gel.py
```

That command rewrites the generated curved/flat files; inspect `git diff` after
running it. Normal setup does not need regeneration or Gmsh. The legacy box is
reference-only. Headless captured RGB uses EGL; check GPU driver support on the
new server. Desktop Studio instead requires an X11 display.

## Replay recorded cases, not a moving recipe

Committed inputs in [reproduction](reproduction):

- `nominal_specs.json`: **21 recorded passing configurations**—19 task/variant
  slots, with the old hook replaced by the new loaded box, plus two extra loaded
  box pose/load checks. Guides and pose pushing use their repaired settings.
- `negative_specs.json`: #87, the rejected 10° rolled loaded-box grasp. It reaches
  the pull distance but exceeds 15° relative rotation; it is not an expert demo.
- `expected_results.json`: original spec/metrics and SHA-256 hashes of source
  HDF5/videos. Original evidence is not embedded; paths are campaign-relative.

Start with a single full flat-scrape case (`--limit 1`), then inspect it:

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python experiments/tool_use_pilot/benchmark.py --output experiments/tool_use_pilot/output/server_check --phase nominal --specs experiments/tool_use_pilot/reproduction/nominal_specs.json --limit 1 --workers 1 --render
```

To replay all 21, omit `--limit 1`, choose a **new** output/phase and set an
appropriate `--workers` count (12 was used for earlier CPU batches). The full
set is not a quick smoke check. The negative replay must use the negative file
in a separate phase; do not concatenate it into imitation data. Episode IDs can
be reassigned by the campaign ledger; match configuration, not only numeric ID.

This selected set is a reproducibility package, not an expansion of the approved
randomization envelope. Old revision-21 baseline/`pose_transfer` recipes remain
historical or candidate recipes and are not automatically release presets.

Check full completion, geometry/RGB, task-specific outcome, grasp retention and
controller behavior against the saved evidence. Compare trajectories with
reasonable tolerances after matching runtime/solver parameters. Do not require
new HDF5 byte hashes to equal the originals: metadata, paths and platform
floating-point details can differ. Original hashes identify retained artifacts.
Force/torque RMSE is diagnostic, never an episode acceptance gate.

## Verification and transfer boundary

Full test discovery also imports the optional learned-probe modules. Install
`pytest`, `ruff`, scikit-learn and CPU Torch from
`requirements-probes.txt` (Torch from its official CPU wheel index), or use an
existing compatible probe environment. They are not required for physics-only
collection. The two CUDA encoder tests skip in a CPU environment.

Tests:

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s experiments/tool_use_pilot -p 'test_*.py'
.venv/bin/python -m pytest experiments/gelsight_mini_contact_validation/test_asset_reproduction.py experiments/gelsight_mini_contact_validation/test_soft_gels.py -q
.venv/bin/ruff check experiments/tool_use_pilot experiments/gelsight_mini_contact_validation/scfields_assets.py --select F,E9
```

Validated here on 11 September: a new cache fetched all 27 inputs from the pinned
revision; **all 27 collision and all 27 surface meshes regenerated byte-for-byte**
against the existing experiment assets. Tests cover immutable-download checks,
corrupt-cache preservation, no-network repeat use, relocation, generation drift,
path containment and frozen replay spec compatibility.
The curved and matched-flat gel meshes and their metadata also regenerated
byte-for-byte. New collection snapshots include the asset lock and simulation
requirements alongside task/controller/gel source.

A fresh checkout using a relocated generated asset cache passed 113 pilot tests
(two CUDA tests skipped) and 29 asset/gel/collector tests. A 10-step installation
check loaded the pinned scraper, initialized both `(7,9,3)` force fields and
rendered EGL RGB from the fresh checkout. It reused the installed wheel runtime;
it was not a clean dependency installation or a new full episode qualification.

Assets/specs alone do not close the scientific gates: action/observation time
alignment, current-task counterfactual dispatch, input-oracle exclusion, grouping,
cross-simulator physics calibration and a small learning/transfer pilot remain
necessary before large collection. Use the handoff; do not reintroduce old return
motions, flat peeling or tactile-wrench eligibility gates during porting.

No remote Git push is implied by these local commits. Push to the user's own
configured remote, or transfer the branch via a Git bundle. The official upstream
remote is not an appropriate destination for private experiment commits without
explicit intent. No outputs were deleted by this packaging work.
