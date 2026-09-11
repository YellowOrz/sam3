from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from sam3.model.spatial_mask_adapter import SpatialMaskAdapter
from scripts.residual_ddp_runtime import capture_rng_state
from scripts.spatial_training_state import FORMAT
from scripts.evaluate_spatial_realsense import selected_checkpoint


class SpatialEvaluationTest(unittest.TestCase):
    def setUp(self):
        model = SpatialMaskAdapter()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=0)
        model(torch.randn(1, 256, 3, 3)).square().mean().backward()
        optimizer.step()
        self.state = dict(format=FORMAT, step=1, last_validation_step=1, best_boundary=.5,
            config=dict(epochs=2, steps_per_epoch=1, world_size=1, learning_rate=.0001,
                        base_sha256='b', tokenizer_sha256='t'),
            adapter=model.state_dict(), optimizer=optimizer.state_dict(), rank_rng=[capture_rng_state('cpu')])

    def load(self, state, epoch=1, base='b'):
        with patch('scripts.evaluate_spatial_realsense.torch.load', return_value=state):
            return selected_checkpoint(Path('synthetic.pt'), epoch, base, 't')

    def test_explicit_validated_epoch(self):
        self.assertEqual(self.load(self.state)['step'], 1)

    def test_reject_partial_unvalidated_or_wrong_base(self):
        with self.assertRaises(ValueError): self.load(self.state, epoch=2)
        with self.assertRaises(ValueError): self.load(self.state, base='wrong')
        state = deepcopy(self.state)
        state['last_validation_step'] = None
        with self.assertRaises(ValueError): self.load(state)

    def test_reject_old_token_format(self):
        state = deepcopy(self.state)
        state['format'] = 'sam3-output-residual-ddp-v1'
        with self.assertRaises(ValueError): self.load(state)
