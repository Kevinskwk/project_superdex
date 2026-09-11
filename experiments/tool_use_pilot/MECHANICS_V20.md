# Mechanics repair before diversification — revision 20

The `transfer` recipes now select revision 20. Revision-19 recordings and their
source snapshots remain historical evidence, not verification of these changes.

- The closed sled arch has only one active layout. Mirrored historical episodes
  remain readable but are not an additional benchmark task.
- Hook pulling is aligned with jaw closing. The grasp cross-section stays aligned
  with the gels; its transition into the working section is continuous. The
  working section and sled are 20 mm lower to clear the finger housings. Size and
  angle randomization are deliberately deferred.
- `levering/spatula_lift` replaces the elaborate gravity-flap setup in the active
  recipe. A straight handled, beveled spatula scoops a free 70 mm-diameter, 8 mm
  thick, 50 g rigid pancake surrogate from a flat surface and lifts it. The
  surrogate starts with a 20 mm edge overhang, while its centre remains 15 mm
  inside the support. Entry is below the overhang, followed by a small edge lift,
  sliding under, and the final lift. The earlier flush-surface entry pushed the
  rigid surrogate instead of scooping it and is not a successful demonstration.
  This edge-assisted entry is narrower than general scooping from a pan. There is
  no hinge, fulcrum, weld, attachment or object-pose controller. This is scoop/lift,
  not deformable food, airborne flipping or completed plate serving.
  The grasp section is 24 mm high and the grasp is 35 mm farther along the handle
  than the first prototype. This improves gel coverage and reduces the unsupported
  moment arm; the tool remains 80 g and the per-finger effort remains 35 N. The
  thin-handle prototype drooped while unsupported and is not the default.
  The entry lift is now 15 mm: the 10 mm trial retained its grasp but left the
  blade about 2 mm below the tabletop under compliance. It hit the table edge
  during insertion and correctly stopped at the unchanged 15 N safety limit.
  Per-body wrench records identify the table, not the pancake, as the obstacle.
  That is a failed trajectory, not evidence of missing tactile causality: its
  force/torque RMSEs were 0.158 N / 0.00054 Nm, with only 0.051 mm peak recorded
  penetration. The separately started rigid friction diagnostic was stopped
  without a conclusion once the full-robot data isolated the clearance issue.
  A separate interface-friction trial preserves effective gel/handle grip at 1.4
  and pancake/table friction at 0.4 while setting blade/pancake friction to 0.15
  (blade/table also 0.4). The engine material factors are derived algebraically
  for its geometric-mean rule, not asserted as calibrated real material values.
  This targets the intended actor-contact pairs; it is not a spatially varying
  per-face material map. New contact graphs require checking the other pairs.
  The 15 mm entry with the original shared friction also failed: it stopped at
  16.76 s on 15.03 N table contact, with no meaningful pancake lift. Grasp drift
  stayed at 2.35 mm / 3.99°, and peak recorded penetration was 0.062 mm. Increasing
  entry height alone therefore did not resolve the insertion path. The separate
  lower-friction trial must be assessed independently, not presumed successful.
- Old turning had **no angular end stop**. Finishing the commanded trajectory was
  not a mechanical stall. The new rotor has a 90-degree native joint limit with
  finite stiffness 2 Nm/rad and damping 0.02 Nms/rad. The 100-degree wrist command
  gently loads that stop through arm and gel compliance. The friction setting is
  0.04 Nm instead of 0.015 Nm. These are simulation parameters, not hardware
  calibration; limit compliance and measured overrun must be reported.
- Curved gels, 7×9×3 force-only observations and fixed physical visualization
  scales remain. No tactile gain is used to manufacture a stronger response.

Keep failed setup attempts separate from selected corrected episodes. A spatula
pass requires a final held lift of at least 30 mm, horizontal drift below 35 mm,
retained grasp, no support escape and no excessive penetration. Its progress is
metres, not the old flap angle. Post-hoc audits also check pancake tilt.

Verification output: `output/mechanics_v20`. Do not treat this development folder
as an accepted dataset without the post-audit index. Broader diversity and any
learned tactile benefit evaluation remain deferred until these mechanics pass.

## Completed checks (9 September 2026)

**Status: hook and all three turning modes pass; spatula scoop/lift does not.**
No further spatula iterations or diversification were started after the final
two clearance/friction trials. See the [selected verification report](output/mechanics_v20/VERIFICATION_REVIEW.md)
and [captured RGB/tactile videos](output/mechanics_v20/representative_videos).
The failed spatula video is explicitly labelled `NOT_PASSED`.

The lower-friction trial also stopped on table contact at 16.88 s / 15.04 N.
Pancake lift was only 0.010 mm; maximum tool drift was 2.46 mm / 5.89°.
Peak pancake contact was 0.214 N, versus 15.04 N on the table. Recorded peak
penetration was 0.065 mm. Force/torque RMSE was 0.159 N / 0.000542 Nm: the field
responded to the unintended contact, but that does not make the task successful.
Lowering blade-interface friction while preserving gel grip did not solve entry.
The current spatula implementation remains a development candidate, not a
qualified benchmark variant. Further work would require revising the approach
geometry and compliance-aware blade clearance; this needs user direction before
another round. A rigid surrogate on a square table edge is also a limited model
of actual pancake serving, so more tuning here is not automatically worthwhile.

Verification: 91 unit tests passed, two optional tests skipped; lint and diff
whitespace checks passed. Sampled actual housing/tool/fixture clearance checks
found no overlap for the five selected episodes, including the failed spatula.
This does not override its safety abort or imply a swept-volume certificate.

The corrected hook and all three turning modes pass full nominal rollouts and
the sampled tool/environment geometry audit. Hook travel reaches 49.23 mm;
maximum grasp movement is 1.17 mm / 0.66°, versus 2.40 mm / 5.02° in the previous
normal layout. This compares repaired layouts, not an isolated rotation sweep.
The saved initial housing poses give pull/closing-axis absolute cosine 0.9998
(previously 0.0219), confirming the intended jaw-aligned pull.

| Turning mode | Final angle | Maximum grasp movement (mm / deg) | Force RMSE (N) | Torque RMSE (Nm) |
|---|---:|---:|---:|---:|
| Spring | 94.09° | 0.77 / 2.68 | 0.153 | 0.00356 |
| Friction | 94.45° | 0.76 / 2.47 | 0.162 | 0.00379 |
| Detent | 94.40° | 0.76 / 2.40 | 0.155 | 0.00366 |

The native rotor reports one limit constraint with bounds [0, π/2]. Its finite
stiffness produces approximately 4.1–4.5° of loaded overrun; this is a compliant
stop approximation, not a perfectly rigid 90° stop or calibrated printed fixture.
The existing 90±5° task criterion is unchanged. Wrench agreement is diagnostic.
Clearance review uses the detailed registered HydroShear housing visuals for
tool/housing checks. The saved conservative housing proxies intentionally overlap
the gel region and their tool contacts are disabled; treating those proxy volumes
as the visible housing would produce false overlap alarms. Future handle/angle
diversity must therefore check the actual housing geometry as well.

Spring final-hold contact torque rises from 0.0316 to 0.176 Nm; the force-only
reconstruction is 0.175 Nm. Marker Δforce RMS rises from 0.075 to 0.318 N/cell.
Friction turning with the new load/stop holds 0.155 Nm, reconstructed as 0.154 Nm.
Reference is the settled 3–4 s grasp, with fixed physical visualization scales.
These force-field changes are not a learned tactile-benefit result. The matched
load/stop controls separate their effects more carefully than this old/new pair.

### Matched resistance × stop controls

Same 100° wrist command and geometry; every control completed with a retained
grasp. No-stop endpoints (~97°) correctly fail the unchanged 90±5° task goal,
but remain physically valid diagnostic trials.

| Friction setting | Stop | Mid-turn contact torque (Nm) | Hold contact torque (Nm) | Hold marker Δforce RMS (N/cell) |
|---|---|---:|---:|---:|
| 0.015 Nm | absent | 0.01568 | 0.00029 | 0.05098 |
| 0.015 Nm | present | 0.01559 | 0.15520 | 0.28315 |
| 0.040 Nm | absent | 0.04120 | 0.00185 | 0.05116 |
| 0.040 Nm | present | 0.04110 | 0.15478 | 0.28259 |

Higher resistance increases moving torque by about 2.6×; the stop creates the
sustained hold load and about 5.5× greater force-field change. Pure torque need
not create a large net force: spatial pressure/shear differences matter. The
no-stop inferred torque retains a roughly 4 mNm residual; these are not exact
or independently calibrated real measurements. See
[the matched-control plot](output/mechanics_v20/matched_stop_controls.png).

The 16-iteration stop check also passes (94.453° final, versus 94.452° with 8).
Across active frames the contact-wrench differences are 0.0155 N / 0.00150 Nm
RMSE, and maximum angle difference is 0.139°. Both native runs report `STOPPED`,
not `CONVERGED`; this is sensitivity evidence, not a convergence certificate.
FP32 / 8 iterations remains the efficient default for these bounded checks.

## Re-run the repaired subset

From the repository root, use a new output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python experiments/tool_use_pilot/benchmark.py --output experiments/tool_use_pilot/output/mechanics_followup --recipe transfer --phase nominal --families hook levering turning --workers 3 --render
```

`--recipe transfer_stop_controls` selects the matched turning controls. The older
`canonical` recipes and revision-18/19 source snapshots are historical protocols,
not the revised mechanics selection. Combined insert–turn and full randomized
qualification are not included in this bounded repair verification.
