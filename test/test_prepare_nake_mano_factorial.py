import tempfile
import unittest
from pathlib import Path
from scripts.eval import prepare_nake_mano_factorial as p


class PrepareNakeFactorialTests(unittest.TestCase):
    def test_unique_active_instances_preserve_source_frames(self):
        box = {'boxes':[[.1,.2,.3,.4]]}
        prompts,sources,active = p.merge_unique([
            ('one.npz',[0,3],{0:box,3:box}),('two.npz',[1,2],{1:box})])
        self.assertEqual(set(prompts),{0,1,3})
        self.assertEqual(active,{0,1,2,3})
        self.assertEqual(sources[1],'two.npz')

    def test_overlap_rejected_even_without_a_usable_projected_box(self):
        with self.assertRaises(ValueError):
            p.merge_unique([('one',[3],{}),('two',[3],{})])

    def test_orphan_and_multiple_boxes_rejected(self):
        box={'boxes':[[.1,.1,.2,.2]]}
        for parts in ([('one',[],{2:box})], [('one',[2],{2:{'boxes':box['boxes']*2}})]):
            with self.assertRaises(ValueError):p.merge_unique(parts)

    def test_excluded_context_does_not_shift_stride_anchor(self):
        self.assertEqual(p.sample_and_exclude(p.EXCLUDED[0],8),([3,6],[0]))
        self.assertEqual(p.sample_and_exclude('other',8),([0,3,6],[]))

    def test_source_path_containment(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for relative in ('../elsewhere','/etc/passwd'):
                with self.assertRaises(ValueError):p.local_file(root,relative)


if __name__ == '__main__':unittest.main()
