"""Package verified, captured RGB+tactile movies; never substitute a pose replay."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import imageio_ffmpeg

from benchmark import PROFILES, SURFACES
from benchmark_geometry import episode_eligibility
from benchmark_report import summarize, geometry_audit


def expected_variants():
    return (
        {("surface", v) for v in SURFACES}
        | {("insertion", v) for v in PROFILES}
        | {("hook", v) for v in ("normal", "mirrored")}
        | {("turning", v) for v in ("spring", "friction", "detent")}
        | {("composite", "key_sequence")}
    )


def validate_coverage(rows):
    keys = [(r["family"], r["variant"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected_variants():
        raise ValueError(
            "Selection must contain exactly one of every active variant, including composite"
        )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package(root, selection, output, review=False):
    root = root.resolve()
    chosen = json.loads(selection.read_text())
    if not review:
        validate_coverage(chosen)
    elif not chosen or len({(r["family"], r["variant"]) for r in chosen}) != len(
        chosen
    ):
        raise ValueError(
            "Review selection must be nonempty with unique family/variant pairs"
        )
    if output.exists():
        raise FileExistsError(
            "Use a new gallery directory; never overwrite reviewed media"
        )
    rows = []
    for item in chosen:
        source = (root / item["episode"]).resolve()
        if not source.is_relative_to(root):
            raise ValueError("Episode must be inside the supplied campaign root")
        record = summarize(source)
        spec, metrics = record["spec"], record["metrics"]
        if (spec["family"], spec["variant"]) != (item["family"], item["variant"]):
            raise ValueError(f"Selection label does not match {source}")
        accepted = episode_eligibility(metrics)["imitation_eligible"]
        geometry = geometry_audit(source) if review else None
        if geometry is not None:
            accepted &= geometry["passed"]
        if not accepted and not review:
            raise ValueError(f"Not an accepted successful full episode: {source}")
        movie = source.with_name(source.stem + "_rgb_tactile.mp4")
        if not movie.is_file():
            raise FileNotFoundError(f"Actual captured movie required: {movie}")
        stream = imageio_ffmpeg.read_frames(str(movie))
        try:
            meta = next(stream)
        finally:
            stream.close()
        # Decode every frame and fail on corrupt packets, not just a valid header.
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-v",
                "error",
                "-xerror",
                "-i",
                str(movie),
                "-f",
                "null",
                "-",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if abs(meta["duration"] - record["seconds_simulated"]) > 0.15:
            raise ValueError(f"Movie does not cover the recorded duration: {movie}")
        rows.append(
            dict(
                family=spec["family"],
                variant=spec["variant"],
                episode=str(source.relative_to(root)),
                episode_id=spec["episode_id"],
                source_movie=str(movie.relative_to(root)),
                video=f"{spec['family']}__{spec['variant']}"
                + ("__NOT_PASSED" if not accepted else "")
                + ".mp4",
                episode_sha256=sha256(source),
                video_sha256=sha256(movie),
                duration_s=meta["duration"],
                fps=meta["fps"],
                frame_size=meta["size"],
                metrics=metrics,
                sampled_geometry_review=geometry,
                passed_review=bool(accepted),
            )
        )
        print(f"Verified {spec['family']}/{spec['variant']}", flush=True)
    output.mkdir(parents=True)
    for row in rows:
        dest = output / row["video"]
        shutil.copy2(root / row["source_movie"], dest)
        if sha256(dest) != row["video_sha256"]:
            raise RuntimeError(f"Copy checksum mismatch: {dest}")
    payload = dict(
        schema="superdex_representative_gallery_v1",
        campaign=str(root),
        video_source="Original captured simulator RGB + synchronized tactile; byte-identical copies",
        scope="Development review: NOT_PASSED movies are failures or incomplete checks, not qualified demonstrations."
        if review
        else "18 isolated nominal variants plus one composite; feasibility, not robustness qualification",
        episodes=rows,
    )
    (output / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Representative pilot benchmark videos",
        "",
        payload["scope"],
        "",
        payload["video_source"],
        "",
        "| Family / variant | Video | Episode | Duration |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['family']} / {row['variant']} | [{row['video']}]({row['video']}) | {row['episode_id']} | {row['duration_s']:.2f} s |"
        )
    lines += [
        "",
        "Convex scraping is retired. Drawing/peeling are rigid-contact variants, not ink deposition or material removal.",
        "Wrench matching is diagnostic only. Source HDF5 paths, checksums and post-audit metrics are in manifest.json.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path(__file__).with_name("representative_episodes.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--review",
        action="store_true",
        help="Package a labeled development subset, including failed checks; never claim full benchmark coverage",
    )
    args = parser.parse_args()
    package(args.campaign, args.selection, args.output, args.review)
