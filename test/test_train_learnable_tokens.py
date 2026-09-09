import unittest

from scripts.train_learnable_tokens import build_epoch_order


class TrainLearnableTokensTest(unittest.TestCase):
    def test_epoch_order_is_deterministic_and_visits_every_sample_once(self):
        first = build_epoch_order(dataset_size=20, seed=123)
        second = build_epoch_order(dataset_size=20, seed=123)

        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(20)))


if __name__ == "__main__":
    unittest.main()
