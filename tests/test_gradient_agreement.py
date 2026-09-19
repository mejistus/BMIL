import unittest

import torch

from clustercontrast.trainers import project_conflicting_gradients


class GradientAgreementTest(unittest.TestCase):
    def test_conflicting_gradient_is_projected(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = torch.tensor([-1.0, 1.0])
        reference = [torch.tensor([1.0, 0.0])]

        cosine, conflicting = project_conflicting_gradients(
            [parameter], reference
        )

        self.assertTrue(conflicting)
        self.assertLess(cosine, 0.0)
        torch.testing.assert_close(parameter.grad, torch.tensor([0.0, 1.0]))

    def test_aligned_gradient_is_unchanged(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = torch.tensor([1.0, 1.0])
        original = parameter.grad.clone()

        cosine, conflicting = project_conflicting_gradients(
            [parameter], [torch.tensor([1.0, 0.0])]
        )

        self.assertFalse(conflicting)
        self.assertGreater(cosine, 0.0)
        torch.testing.assert_close(parameter.grad, original)


if __name__ == "__main__":
    unittest.main()
