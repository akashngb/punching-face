import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.head_accessories import clean_view,paths_pixels
from scripts.hair_groom import build_hair_groom

class HairEyewearTests(unittest.TestCase):
    def spec(self,present=True):
        polygon=[[.2,.3],[.4,.3],[.45,.35],[.45,.5],[.4,.55],[.2,.55],[.15,.5],[.15,.35]]
        return {'glasses':{'present':present,'confidence':.2},'crops':{'front':[0,0,100,100]},'views':[{'filename':'front','imageLeftLens':polygon,'hairRegions':[[[.1,.1],[.8,.1],[.8,.2]]],'eyewearRegions':[]}]}
    def test_frame_cleanup_preserves_eye_and_skin_detail_inside_clear_lenses(self):
        px=np.full((100,100,4),255,np.uint8);px[:,:,:3]=[170,120,90];px[35:48,23:38,:3]=[35,25,20]
        cleaned,mask,audit=clean_view(px,'front',self.spec(),{},return_details=True)
        self.assertEqual(mask[40,30],0);np.testing.assert_array_equal(cleaned[40,30],px[40,30])
        self.assertGreater(audit['excludedPixels'],50);self.assertTrue(audit['preservedEyePhotographs'])
        self.assertEqual(set(paths_pixels(self.spec()['views'][0],[0,0,100,100])),{'imageLeftLens'})
    def test_rear_views_and_no_glasses_are_preserved(self):
        px=np.full((100,100,4),255,np.uint8)
        np.testing.assert_array_equal(clean_view(px,'rear',self.spec(),{}),px)
        np.testing.assert_array_equal(clean_view(px,'front',self.spec(False),{}),px)
    def test_bald_and_zero_density_never_generate_a_hair_layer(self):
        p=np.zeros((468,3));f=np.array([[0,1,2]])
        self.assertIsNone(build_hair_groom(p,f,{'hair':{'present':False}}))
        self.assertIsNone(build_hair_groom(p,f,{'hair':{'present':True,'type':'bald'}}))
        self.assertIsNone(build_hair_groom(p,f,{'hair':{'present':True,'density':0}}))

if __name__=='__main__':unittest.main()
