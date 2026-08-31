from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from collect_parallel import (
    SCHEMA_VERSION,
    allocate,
    episode_specs,
    validate_hdf5,
    write_hdf5,
)


def test_vt_acwm_hdf5_contract(tmp_path: Path) -> None:
    args = SimpleNamespace(
        episodes=2,
        steps=5,
        point_count=32,
        compression="lzf",
        dt=0.01,
    )
    data = allocate(args.episodes, args.steps, args.point_count)
    data["dones"][:, -1] = True
    data["table_contact_count"][:, -1] = 1
    data["left_contact_count"][:, -1] = 1
    data["right_contact_count"][:, -1] = 1
    specs = episode_specs(args.episodes, 7)
    path = tmp_path / "episodes.hdf5"
    write_hdf5(path, data, specs, args, {"episodes": 2})

    result = validate_hdf5(path, args.episodes, args.steps, args.point_count)
    assert result == {
        "schema_valid": True,
        "episodes_valid": 2,
        "episodes_with_table_contact": 2,
        "episodes_with_bilateral_tactile": 2,
    }
    with h5py.File(path, "r") as stream:
        assert stream.attrs["schema_version"] == SCHEMA_VERSION
        assert len(stream["episodes"]) == 2
        episode = stream["episodes/episode_000000"]
        observations = episode["observations"]
        assert observations["point_cloud"].id == observations["objectpointcloud"].id
        assert observations["tactile_data_left"].id == observations[
            "tactile_force_field_left"
        ].id
        assert observations["objectpointcloud"].shape == (5, 32, 3)
        assert observations["tactile_force_field_left"].shape == (5, 7, 9, 3)
        assert episode["actions"].shape == (5, 6)
        assert np.asarray(episode.attrs["initial_state_hash"]).item()
