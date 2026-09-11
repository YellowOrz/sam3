from copy import deepcopy
from test_spatial_training_state import SpatialStateTests
from scripts.extend_spatial_budget import extend_budget


class BudgetExtensionTests(SpatialStateTests):
    def completed(self):
        state = deepcopy(self.state)
        state['config']['steps_per_epoch'] = 1
        state['config']['epochs'] = 1
        state['last_validation_step'] = 1
        return state

    def test_extension_preserves_training_state(self):
        state = self.completed()
        result = extend_budget(state, 10, self.adapter.state_dict(), 'source-hash')
        self.assertEqual(state['config']['epochs'], 1)
        self.assertEqual(result['config']['epochs'], 10)
        for key in ('adapter', 'optimizer', 'rank_rng'):
            self.assertIs(result[key], state[key])
        self.assertEqual(result['step'], state['step'])

    def test_reject_incomplete_unvalidated_or_reduced(self):
        for state, epochs in [(self.state, 10), (self.completed(), 1)]:
            with self.assertRaises(ValueError):
                extend_budget(state, epochs, self.adapter.state_dict(), 'hash')
        state = self.completed()
        state['last_validation_step'] = 0
        with self.assertRaises(ValueError):
            extend_budget(state, 10, self.adapter.state_dict(), 'hash')
