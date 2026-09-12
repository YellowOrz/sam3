from copy import deepcopy
from test_spatial_training_state import SpatialStateTests
from scripts.spatial_training_state import transfer_stage, validate_state
from scripts.residual_ddp_runtime import torchrun_ranks


class StageTransferTests(SpatialStateTests):
    def source_and_config(self):
        source = deepcopy(self.state)
        source['config'].update(epochs=1, steps_per_epoch=1,
            annotation_hashes={'train': 'nake', 'val': 'dex-val'})
        source['last_validation_step'] = 1
        config = dict(source['config'], epochs=8, steps_per_epoch=5,
            annotation_hashes={'train': 'mixed', 'val': 'dex-val'},
            stage_source_sha256='hash', optimizer_step_offset=1)
        return source, config

    def test_preserves_optimizer_and_rng_resets_sampler(self):
        source, config = self.source_and_config()
        result = transfer_stage(source, config, self.adapter.state_dict(), 'hash')
        self.assertEqual(result['step'], 0)
        self.assertIsNone(result['last_validation_step'])
        for key in ('adapter', 'optimizer', 'rank_rng'):
            self.assertIs(result[key], source[key])
        validate_state(result, config, self.adapter.state_dict())
        self.assertEqual(source['step'], 1)

    def test_reject_unapproved_changes(self):
        for key, value in [('learning_rate', .1), ('world_size', 2),
                           ('optimizer_step_offset', 0), ('stage_source_sha256', 'wrong'),
                           ('annotation_hashes', {'train': 'mixed', 'val': 'other'})]:
            source, config = self.source_and_config()
            config[key] = value
            with self.assertRaises(ValueError):
                transfer_stage(source, config, self.adapter.state_dict(), 'hash')

    def test_reject_partial(self):
        source, config = self.source_and_config()
        source['last_validation_step'] = 0
        with self.assertRaises(ValueError):
            transfer_stage(source, config, self.adapter.state_dict(), 'hash')

    def test_cpu_preflight_must_plan_same_world_as_parent(self):
        source, config = self.source_and_config()
        source['config']['world_size'] = 2
        source['rank_rng'] = source['rank_rng'] * 2
        config['world_size'] = torchrun_ranks({})[2]
        with self.assertRaisesRegex(ValueError, 'world_size'):
            transfer_stage(source, config, self.adapter.state_dict(), 'hash')
        config['world_size'] = torchrun_ranks(
            {'RANK': '0', 'LOCAL_RANK': '0', 'WORLD_SIZE': '2'})[2]
        result = transfer_stage(source, config, self.adapter.state_dict(), 'hash')
        self.assertEqual(len(result['rank_rng']), 2)
