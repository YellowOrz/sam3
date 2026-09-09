import unittest

import torch

from sam3.model.vitdet import Mlp


class VitdetMlpTest(unittest.TestCase):
    def test_training_path_supports_backward(self):
        mlp = Mlp(in_features=4, hidden_features=8)
        inputs = torch.randn(2, 3, 4, requires_grad=True)

        output = mlp(inputs)
        output.sum().backward()

        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertIsNotNone(mlp.fc1.weight.grad)

    def test_inference_path_matches_second_linear_dtype(self):
        mlp = Mlp(in_features=4, hidden_features=8)
        inputs = torch.randn(2, 3, 4)

        with torch.no_grad():
            output = mlp(inputs)

        self.assertEqual(output.dtype, mlp.fc2.weight.dtype)


if __name__ == "__main__":
    unittest.main()
