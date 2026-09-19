import math
import unittest

import torch

from clustercontrast.models.cm import ClusterMemory


def make_memory(proxy_labels):
    memory = ClusterMemory.__new__(ClusterMemory)
    torch.nn.Module.__init__(memory)
    memory.proxy_pseudo_labels = torch.tensor(proxy_labels, dtype=torch.long)
    return memory


class IdentityMassContrastTest(unittest.TestCase):
    def test_duplicate_proxy_does_not_add_identity_mass(self):
        single = make_memory([0, 1])
        dual = make_memory([0, 1, 1])
        single_score = torch.tensor([[1.5, 0.4]], requires_grad=True)
        dual_score = torch.tensor([[1.5, 0.4, 0.4]], requires_grad=True)

        single_logits, _ = single._identity_balanced_logits(single_score)
        dual_logits, counts = dual._identity_balanced_logits(dual_score)

        torch.testing.assert_close(single_logits, dual_logits)
        self.assertEqual(counts.tolist(), [1, 2])

    def test_dual_proxy_uses_log_mean_exp(self):
        memory = make_memory([0, 0, 1])
        score = torch.tensor([[0.2, 1.2, -0.4]], requires_grad=True)
        logits, _ = memory._identity_balanced_logits(score)
        expected = torch.logsumexp(score[0, :2], dim=0) - math.log(2.0)

        torch.testing.assert_close(logits[0, 0], expected)
        torch.testing.assert_close(logits[0, 1], score[0, 2])

    def test_loss_is_finite_and_differentiable(self):
        memory = make_memory([0, 0, 1, 2, 2])
        score = torch.randn(4, 5, requires_grad=True)
        proxy_targets = torch.tensor([0, 2, 3, 4])

        loss = memory.get_multi_pos_loss(score, proxy_targets)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(score.grad).all())


if __name__ == "__main__":
    unittest.main()
