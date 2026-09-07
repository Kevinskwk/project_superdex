"""Audit recorded RGB clips and create a labeled review contact sheet."""

import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    a = p.parse_args()
    paths = sorted(a.folder.glob("verification/*/replay/*_rgb_tactile.mp4"))
    sheet = Image.new("RGB", (1200, 330 * len(paths)), "white")
    draw = ImageDraw.Draw(sheet)
    audit = []
    for row, path in enumerate(paths):
        verification = json.loads(
            (path.parent.parent / "verification.json").read_text()
        )
        spec = verification["replay"]["spec"]
        m = verification["replay"]["branches"][0]
        reader = imageio.get_reader(path)
        meta = reader.get_meta_data()
        frames = reader.count_frames()
        status = (
            "VALID SUCCESS"
            if m["physical_valid"] and m["task_success"]
            else "REJECTED: not a success demonstration"
        )
        stds = []
        for col, fraction in enumerate((0.20, 0.55, 0.80)):
            index = min(frames - 1, round(frames * fraction))
            rgb = reader.get_data(index)
            stds.append(float(rgb[:, :800].std()))
            sheet.paste(
                Image.fromarray(rgb[:, :800]).resize((400, 300)),
                (col * 400, row * 330 + 30),
            )
            draw.text(
                (col * 400 + 5, row * 330 + 3),
                f"{path.parent.parent.name} t={index / meta['fps']:.1f}s",
                fill="black",
            )
            draw.text(
                (col * 400 + 5, row * 330 + 16),
                status,
                fill="green" if m["physical_valid"] else "red",
            )
        reader.close()
        audit.append(
            dict(
                video=str(path.relative_to(a.folder)),
                fps=meta["fps"],
                frames=frames,
                expected_frames=round(spec["duration"] * 20),
                nonblank=min(stds) > 10,
                frame_count_matches=frames == round(spec["duration"] * 20),
                physical_valid=m["physical_valid"],
                task_success=m["task_success"],
                exact_replay=verification["exact"],
            )
        )
    sheet.save(a.folder / "video_contact_sheet.png")
    (a.folder / "media_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
