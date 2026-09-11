import unittest
import numpy as np
from scripts.prepare_nakehand_manual_review import comparison_sheet,sample_records


class ManualReviewTest(unittest.TestCase):
    def counts(self):
        return {**{f'nakehandego/{i}':100 for i in range(4)},**{f'nakehandexo/{i}':100 for i in range(2)}}

    def test_draw_reproducible_excluded_and_per_recording_stratified(self):
        excluded={(name,i) for name in self.counts() for i in range(50)}
        selected,candidates=sample_records(self.counts(),excluded,20260910)
        self.assertEqual((selected,candidates),sample_records(self.counts(),excluded,20260910))
        self.assertEqual(len(selected),10)
        self.assertEqual([s['review_id'] for s in selected],[f'{i:02d}' for i in range(1,11)])
        self.assertTrue(all(s['frame_index']>=50 for s in selected))
        for name in self.counts():
            self.assertEqual(sum(s['recording_id']==name for s in selected),2 if 'ego' in name else 1)

    def test_short_candidate_pool_fails_instead_of_reusing_old_sample(self):
        excluded={(name,i) for name in self.counts() for i in range(100)}
        with self.assertRaises(ValueError): sample_records(self.counts(),excluded)

    def test_scientific_sheet_preserves_three_pixel_panels_including_empty(self):
        rgb=np.arange(36,dtype=np.uint8).reshape(3,4,3)
        left=np.zeros((3,4),dtype=np.uint8)
        right=left.copy();right[1,2]=255
        sample={'review_id':'01','recording_id':'nakehandego/one','frame_index':4}
        sheet=np.asarray(comparison_sheet(rgb,left,right,sample))
        self.assertTrue(np.array_equal(sheet[74:,:4],rgb))
        self.assertTrue(np.array_equal(sheet[74:,4:8],np.repeat(left[:,:,None],3,axis=2)))
        self.assertTrue(np.array_equal(sheet[74:,8:],np.repeat(right[:,:,None],3,axis=2)))


if __name__=='__main__': unittest.main()
