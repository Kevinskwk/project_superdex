# Artifact retention — 7 September 2026 work log

Cleanup executed on 8 September SGT; the experiment progress is logged under
7 September as requested. No physics, controller, sensing, or validity threshold
was changed by cleanup.

## Retained locally

| Output directory | Why it remains |
|---|---|
| `final_pilot` | Original four-task final dataset, branch/replay evidence, plots and RGB videos |
| `decision_v2` | Earlier grip/material/decision screens, including important negative findings |
| `hook_repair_final` | Repaired hook geometry and full recovery-branch verification/videos |
| `hook_tactile_observability` | Noisy/occluded PCD sensing study and held-out predictions |
| `key_stages_split` | Separate insertion/turning verification and representative RGB/tactile videos |
| `key_tactile_observability` | Current sensing data, raw failed signed-offset/return trials, qualified follow-up, predictions and reports |

All six directories retain their existing contents and source snapshots.
The Point-M2AE checkout and checkpoint remain local but ignored, as do all outputs.
SCFields assets remain ignored by the repository root.

## Removed / recoverability

38 superseded output directories were removed after archiving their reports,
JSON metrics/configurations, plots, logs and Python snapshots into
`output/superseded_evidence_20260907.tar.gz`. Archive members were verified against
SHA-256 hashes before deletion. All 643 external HDF5 links encountered in retained
datasets were checked for existing targets outside the deletion set.

`CLEANUP_20260907.json` records exact deleted paths/sizes, archived-file hashes and
the archive digest. Superseded raw HDF5, videos, model weights and feature caches
are **not recoverable from this archive**; rerunning would be required. Reports
inside the historical archive describe the original runs, not current retention.

Retired `assemble.py` (one-off cross-revision copying), `findings.py` (the original
study-specific narrative generator), and `visual_audit.py` (one-off montage) are
also archived. Their completed outputs remain in `final_pilot`. Current hook/key
reports and visual/mesh audits remain executable. Keep the six small regression
test modules: they protect active geometry, wrench transport, data splits and
pretrained-encoder contracts; they are not disposable smoke experiments.

## Repeatable verification

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s experiments/tool_use_pilot -p 'test_*.py'
.venv/bin/ruff check --select F,E9 experiments/tool_use_pilot --exclude vendor --exclude output
```

No large-scale recollection is needed for cleanup. The current experiment reports
remain local evidence, not claims of real-sensor calibration or cross-domain transfer.
