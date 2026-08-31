#!/usr/bin/env python3
"""Recompute per-episode campaign metrics from completed HDF5 shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from campaign_specs import EpisodeSpec
from fidelity_campaign import episode_metrics


OBSERVATION_KEYS = (
    "direct_gel_wrench", "field_gel_wrench",
    "tactile_force_field_left", "tactile_force_field_right",
    "extrinsic_contact_wrench", "dynamic_inferred_extrinsic_wrench",
    "environment_contact_count",
    "direct_gel_centroid_left", "direct_gel_centroid_right",
    "field_gel_centroid_left", "field_gel_centroid_right",
    "direct_environment_cop", "inferred_environment_cop",
    "ee_pose", "tool_pose", "left_contact_count", "right_contact_count",
)


def spec_from_group(group: h5py.Group) -> EpisodeSpec:
    fields = EpisodeSpec.__dataclass_fields__
    return EpisodeSpec(**{
        key: group.attrs[key].item() if isinstance(group.attrs[key], np.generic) else group.attrs[key]
        for key in fields
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    shard_paths = sorted((args.campaign / "campaign_shards").glob("shard_*.hdf5"))
    if not shard_paths:
        raise FileNotFoundError("no campaign shards found")
    for shard_path in shard_paths:
        metrics: dict[int, dict[str, object]] = {}
        with h5py.File(shard_path, "r+") as stream:
            dt = float(stream.attrs["sample_period_s"])
            for group in stream["episodes"].values():
                spec = spec_from_group(group)
                observations = group["observations"]
                data = {key: observations[key][...][None] for key in OBSERVATION_KEYS}
                data["contact_phase"] = group["contact_phase"][...][None]
                values = episode_metrics(data, 0, spec, dt)
                metric_group = group["metrics"]
                for key in list(metric_group.attrs):
                    del metric_group.attrs[key]
                for key, value in values.items():
                    metric_group.attrs[key] = value
                for key, metric_key in (
                    ("physical_valid", "physical_valid"),
                    ("dense_fidelity_pass", "dense_pass"),
                    ("environment_fidelity_pass", "environment_pass"),
                    ("grasp_retained", "grasp_retained"),
                ):
                    path = f"labels/{key}"
                    if path in group:
                        group[path][...] = np.bool_(values[metric_key])
                    else:
                        group["labels"].create_dataset(key, data=np.bool_(values[metric_key]))
                metrics[spec.episode_id] = values
        sidecar_path = shard_path.with_suffix(".json")
        sidecar = json.loads(sidecar_path.read_text())
        sidecar["metrics"] = metrics
        sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
        print(f"updated {len(metrics)} episodes in {shard_path.name}", flush=True)


if __name__ == "__main__":
    main()
