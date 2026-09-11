from copy import deepcopy
import unittest
import torch
from sam3.model.spatial_mask_adapter import SpatialMaskAdapter
from scripts.spatial_training_state import FORMAT, validate_state
from scripts.residual_ddp_runtime import capture_rng_state


class SpatialStateTests(unittest.TestCase):
    def setUp(self):
        self.adapter = SpatialMaskAdapter(8, 4)
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=.0001, weight_decay=0)
        self.adapter(torch.randn(2, 8, 5, 5)).square().mean().backward()
        optimizer.step()
        self.config = dict(epochs=2, steps_per_epoch=3, world_size=1, learning_rate=.0001)
        self.state = dict(format=FORMAT, config=deepcopy(self.config), step=1,
            adapter=deepcopy(self.adapter.state_dict()), optimizer=optimizer.state_dict(),
            rank_rng=[capture_rng_state('cpu')], last_validation_step=0, best_boundary=.4)

    def check(self, state):
        return validate_state(state, self.config, self.adapter.state_dict())

    def test_valid_and_optimizer_resume(self):
        self.assertEqual(self.check(self.state), 1)
        restored = SpatialMaskAdapter(8, 4)
        restored.load_state_dict(self.state['adapter'])
        opt = torch.optim.AdamW(restored.parameters(), lr=.0001, weight_decay=0)
        opt.load_state_dict(self.state['optimizer'])
        self.assertEqual(len(opt.state), 6)

    def test_reject_token_and_changed_config(self):
        for field, value in [('format', 'sam3-output-residual-ddp-v1'), ('config', {})]:
            state = deepcopy(self.state)
            state[field] = value
            with self.assertRaises(ValueError): self.check(state)

    def test_bad_progress_rng_and_validation(self):
        for field, value in [('step', True), ('step', 7), ('rank_rng', []),
                             ('last_validation_step', 1), ('best_boundary', float('nan'))]:
            state = deepcopy(self.state)
            state[field] = value
            with self.assertRaises(ValueError): self.check(state)

    def test_bad_parameter_and_optimizer(self):
        state = deepcopy(self.state)
        state['adapter']['up.bias'][0] = float('nan')
        with self.assertRaises(ValueError): self.check(state)
        state = deepcopy(self.state)
        state['optimizer']['state'] = {}
        with self.assertRaises(ValueError): self.check(state)
        state = deepcopy(self.state)
        state['optimizer']['param_groups'][0]['lr'] = .1
        with self.assertRaises(ValueError): self.check(state)

    def test_wrong_moment_shape_and_step(self):
        for field, value in [('exp_avg', torch.zeros(1)), ('step', torch.tensor(2.))]:
            state = deepcopy(self.state)
            state['optimizer']['state'][0][field] = value
            with self.assertRaises(ValueError): self.check(state)
