"""GPU integration checks; CPU simulation environments skip encoder execution."""

import unittest
import numpy as np
import torch
from decision_probes import task_gate
from point_m2ae_probe import FrozenPointM2AE, group, fps


class GateTests(unittest.TestCase):
    def test_invalid_sibling_cannot_be_hidden(self):
        rows = [
            dict(
                prefix_valid=True,
                siblings=[
                    dict(name="a", valid=True, success=True),
                    dict(name="b", valid=False, success=False),
                ],
            )
        ]
        self.assertFalse(task_gate(rows)["passed"])
        self.assertEqual(task_gate(rows)["clean_anchors"], 0)

    def test_constant_action_is_not_state_dependent(self):
        rows = [
            dict(
                prefix_valid=True,
                siblings=[
                    dict(name="a", valid=True, success=True),
                    dict(name="b", valid=True, success=False),
                ],
            )
            for _ in range(8)
        ]
        self.assertFalse(task_gate(rows)["passed"])
        for row in rows[:4]:
            row["siblings"][0]["success"] = False
            row["siblings"][1]["success"] = True
        self.assertTrue(task_gate(rows)["passed"])


@unittest.skipUnless(
    torch.cuda.is_available(), "requires the separate CUDA probe environment"
)
class PretrainedGPU(unittest.TestCase):
    def test_grouping_matches_brute_force(self):
        points = torch.tensor(
            [[[0.0, 0, 0], [1, 0, 0], [3, 0, 0], [5, 0, 0]]], device="cuda"
        )
        self.assertEqual(fps(points, 2).tolist(), [[0, 3]])
        neighbors, centers, indices = group(points, 2, 2)
        self.assertEqual(indices.tolist(), [0, 1, 3, 2])
        torch.testing.assert_close(
            neighbors, points[:, indices.reshape(2, 2)] - centers[:, :, None]
        )

    def test_pretrained_encoder_frozen_head_trainable(self):
        torch.manual_seed(7)
        model = FrozenPointM2AE().cuda()
        points = torch.randn(2, 1024, 3, device="cuda") * 0.2
        before = {k: v.clone() for k, v in model.state_dict().items()}
        a = model(points)
        b = model(points)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        head = torch.nn.Linear(384, 1).cuda()
        old = head.weight.detach().clone()
        optimizer = torch.optim.Adam(head.parameters(), lr=0.001)
        for _ in range(3):
            optimizer.zero_grad()
            loss = head(a).square().mean()
            loss.backward()
            optimizer.step()
        self.assertFalse(torch.equal(old, head.weight))
        self.assertTrue(
            all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
        )
        self.assertEqual(tuple(a.shape), (2, 384))
        self.assertTrue(np.isfinite(a.cpu().numpy()).all())


if __name__ == "__main__":
    unittest.main()
