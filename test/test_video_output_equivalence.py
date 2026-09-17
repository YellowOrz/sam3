import copy
import json
from pathlib import Path
import tempfile
import unittest
from scripts.eval.video_output_equivalence import SavedOutputEquivalence, sha


class EquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.info = dict(contract={}, plan_sha256='plan', base_sha256='base',
            model_source_sha256={}, mode='text', gpu='gpu', torch_version='version',
            engineering=False, runner_sha256='original-runner')
        self.write('run.json', self.info)
        self.rows = [dict(sequence='a',frame_index=i,method=method,prediction_pixels=i,
            instances=[{'id':1,'score':.75,'rle':{'counts':'a','size':[2,2]}}])
            for i in range(2) for method in ('frame_text','video_text')]
        self.save_rows()
        self.write('a-complete.json',dict(status='complete',frames=2,
            records_sha256=sha(self.root/'a.jsonl')))

    def write(self, path, value):
        (self.root/path).write_text(json.dumps(value))

    def save_rows(self):
        (self.root/'a.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))

    def test_exact_prefix_and_new_frames(self):
        q=SavedOutputEquivalence(self.root,self.info)
        q.begin('a',3)
        for row in self.rows: q.check(row)
        q.check(dict(sequence='a',frame_index=2,method='frame_text'))
        q.finish_sequence()
        q.begin('new',4); q.finish_sequence()
        self.assertEqual(q.summary()['identical_records'],4)

    def test_algorithm_change_is_not_storage_equivalence(self):
        bad=dict(self.info,tracker_policy='successful-recondition-selected-mask-v1',tracker_policy_sha256='patch')
        with self.assertRaisesRegex(ValueError,'Algorithm policy differs'):
            SavedOutputEquivalence(self.root,bad)
        SavedOutputEquivalence(self.root,dict(self.info,tracker_policy='legacy',tracker_policy_sha256=None))

    def test_engineering_subset(self):
        q=SavedOutputEquivalence(self.root,self.info); q.begin('a',1)
        for row in self.rows[:2]: q.check(row)
        q.finish_sequence()
        self.assertEqual(q.summary()['identical_records'],2)

    def test_mask_id_or_score_change_is_not_tolerated(self):
        for key,value in [('id',2),('score',.750001),('rle',{'counts':'b','size':[2,2]})]:
            q=SavedOutputEquivalence(self.root,self.info); q.begin('a',2)
            row=copy.deepcopy(self.rows[0]); row['instances'][0][key]=value
            with self.assertRaisesRegex(ValueError,'output changed'): q.check(row)

    def test_missing_comparison_and_mutation_rejected(self):
        q=SavedOutputEquivalence(self.root,self.info); q.begin('a',2)
        with self.assertRaisesRegex(ValueError,'every executed'): q.finish_sequence()
        self.write('run.json',{})
        with self.assertRaisesRegex(ValueError,'changed during'): q.summary()

    def test_partial_requires_stopped_run_and_pairs(self):
        (self.root/'a-complete.json').unlink()
        q=SavedOutputEquivalence(self.root,self.info)
        with self.assertRaisesRegex(ValueError,'stopped failed'): q.begin('a',2)
        self.write('failure.json',{'error':'OOM'})
        q.begin('a',2)
        self.rows.pop(); self.save_rows()
        with self.assertRaisesRegex(ValueError,'pairs'): q.begin('a',2)

    def test_original_manifest_and_runtime_mismatch(self):
        bad=dict(self.info,gpu='different')
        with self.assertRaisesRegex(ValueError,'Unpaired'): SavedOutputEquivalence(self.root,bad)
        self.rows[0]['prediction_pixels']=42; self.save_rows()
        q=SavedOutputEquivalence(self.root,self.info)
        with self.assertRaisesRegex(ValueError,'completed sequence changed'): q.begin('a',2)


if __name__=='__main__': unittest.main()
