"""Frozen official Point-M2AE encoder with portable, deterministic FPS/KNN.

Only the published encoder class definitions are loaded; reconstruction losses,
training launchers and legacy CUDA extensions are deliberately not imported.
"""

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
EXPECTED_SOURCE_SHA256 = (
    "54e774f1e4421022598feb0d9cadbf2decee2da5a4ae3cea916ac73561c43b36"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "6dd8f7a5993f6965c76772c6049824d33fd3043473f71079b210ebec78a644a3"
)


@torch.no_grad()
def fps(xyz, count):
    b, n, _ = xyz.shape
    if count > n:
        raise ValueError("FPS count exceeds observed/resampled input")
    indices = torch.empty((b, count), dtype=torch.long, device=xyz.device)
    distance = torch.full((b, n), float("inf"), device=xyz.device)
    farthest = torch.zeros(b, dtype=torch.long, device=xyz.device)
    batch = torch.arange(b, device=xyz.device)
    for i in range(count):
        indices[:, i] = farthest
        d = ((xyz - xyz[batch, farthest, None]) ** 2).sum(-1)
        distance = torch.minimum(distance, d)
        farthest = distance.argmax(-1)
    return indices


def group(xyz, count, k):
    batch = torch.arange(xyz.shape[0], device=xyz.device)
    center = xyz[batch[:, None], fps(xyz, count)]
    distances = torch.cdist(center, xyz, compute_mode="donot_use_mm_for_euclid_dist")
    idx = distances.argsort(dim=-1, stable=True)[..., :k]
    neighbors = xyz[batch[:, None, None], idx] - center[:, :, None]
    flat = idx + batch[:, None, None] * xyz.shape[1]
    return neighbors, center, flat.reshape(-1)


def definitions(path, names, namespace):
    module = ast.parse(path.read_text())
    body = [
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name in names
    ]
    if len(body) != len(names):
        raise ValueError(f"upstream class layout changed: {path}")
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)


class FrozenPointM2AE(nn.Module):
    def __init__(self, checkpoint=None, vendor=None):
        super().__init__()
        vendor = Path(vendor or HERE / "vendor/Point-M2AE")
        checkpoint = Path(checkpoint or HERE / "checkpoints/pre-train.pth")
        source_hash = hashlib.sha256(
            (vendor / "models/modules.py").read_bytes()
            + (vendor / "models/Point_M2AE.py").read_bytes()
        ).hexdigest()
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if (
            source_hash != EXPECTED_SOURCE_SHA256
            or checkpoint_hash != EXPECTED_CHECKPOINT_SHA256
        ):
            raise ValueError(
                "Point-M2AE source/checkpoint differs from the pinned verified release"
            )
        ns = dict(torch=torch, nn=nn, np=np, trunc_normal_=nn.init.trunc_normal_)
        definitions(
            vendor / "models/modules.py",
            {"Token_Embed", "Mlp", "Attention", "Block", "Encoder_Block"},
            ns,
        )
        definitions(vendor / "models/Point_M2AE.py", {"H_Encoder"}, ns)
        cfg = SimpleNamespace(
            mask_ratio=0.8,
            encoder_depths=[5, 5, 5],
            encoder_dims=[96, 192, 384],
            local_radius=[0.32, 0.64, 1.28],
            drop_path_rate=0.0,
            num_heads=6,
        )
        self.h_encoder = ns["H_Encoder"](cfg)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)[
            "base_model"
        ]
        state = {
            k.removeprefix("module."): v
            for k, v in saved.items()
            if k.removeprefix("module.").startswith("h_encoder.")
        }
        self.load_state_dict(state, strict=True)
        self.requires_grad_(False)
        self.eval()
        self.checkpoint_sha256 = checkpoint_hash
        self.source_hash = source_hash

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, points):
        if not points.is_cuda:
            raise ValueError("official encoder evaluation path requires CUDA")
        neighborhoods = []
        centers = []
        indices = []
        xyz = points
        for count, k in zip([512, 256, 64], [16, 8, 8]):
            n, c, i = group(xyz, count, k)
            neighborhoods.append(n)
            centers.append(c)
            indices.append(i)
            xyz = c
        tokens, _, _ = self.h_encoder(neighborhoods, centers, indices, eval=True)
        return tokens[-1].mean(1) + tokens[-1].max(1).values


def self_test():
    import json, time

    torch.manual_seed(0)
    model = FrozenPointM2AE().cuda()
    points = torch.randn(2, 1024, 3, device="cuda")
    points = points / points.norm(dim=-1).max(-1).values[:, None, None]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    start = time.perf_counter()
    a = model(points)
    b = model(points)
    torch.cuda.synchronize()
    assert a.shape == (2, 384) and torch.isfinite(a).all()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    print(
        json.dumps(
            dict(
                checkpoint_sha256=model.checkpoint_sha256,
                source_hash=model.source_hash,
                encoder_parameters=sum(p.numel() for p in model.parameters()),
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
                seconds=time.perf_counter() - start,
                peak_vram_mb=torch.cuda.max_memory_allocated() / 2**20,
                deterministic=True,
                frozen=True,
                shape=list(a.shape),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    self_test()
