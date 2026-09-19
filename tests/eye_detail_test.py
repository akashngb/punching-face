"""Eye quality/provenance gates and isolation from photographic skin."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.eye_detail import default_spec,eye_colors,photo_gate,apply_eye_material,scan_eyes,fit_eye_depth,EYES

class EyeDetailTests(unittest.TestCase):
    def setUp(self):
        self.spec={**default_spec(),'photoUsable':True,'confidence':.95,'irisRadius':.2}
        self.candidate={'openingPx':35,'sharpness':100,'glareFraction':0,'lidPolygon':[[10,25],[50,20],[90,25],[90,75],[50,80],[10,75]]}
        yy,xx=np.mgrid[:100,:100]
        self.photo=np.uint8(eye_colors(np.c_[(xx.ravel()-50)/20,(50-yy.ravel())/20],self.spec).reshape(100,100,3)*255)
    def test_only_native_resolved_iris_passes(self):
        self.assertTrue(photo_gate(self.candidate,self.spec,self.photo)[0])
        for spec,candidate,pixels in [
            ({**self.spec,'irisRadius':.08},self.candidate,self.photo),
            (self.spec,{**self.candidate,'openingPx':5},self.photo),
            (self.spec,{**self.candidate,'glareFraction':.2},self.photo),
            (self.spec,{**self.candidate,'sharpness':3},self.photo),
            ({**self.spec,'irisCenter':[.05,.05]},self.candidate,self.photo),
            (self.spec,self.candidate,np.full_like(self.photo,110)),
            ({**self.spec,'photoUsable':False},self.candidate,self.photo),
        ]:self.assertFalse(photo_gate(candidate,spec,pixels)[0])
    def test_eye_material_does_not_change_skin_or_lids(self):
        p=np.zeros((470,3));p[33]=[-.06,.04,0];p[133]=[-.02,.04,0];p[159]=[-.04,.045,0];p[145]=[-.04,.035,0]
        p[263]=[.06,.04,0];p[362]=[.02,.04,0];p[386]=[.04,.045,0];p[374]=[.04,.035,0]
        labels=np.zeros(len(p),np.uint8);labels[468:]=[1,2];texel=np.array([[0,.1,0],[-.04,.04,0],[.04,.04,0]])
        colors=np.full((3,3),.35);details={'eyes':{name:default_spec() for name in ['imageLeft','imageRight']}}
        out,count=apply_eye_material(Path('/unused'),texel,np.array([0,1,2]),p,labels,colors,details)
        np.testing.assert_array_equal(out[0],colors[0]);self.assertEqual(count,2);self.assertLess(out[1].mean(),.05)
        self.assertTrue(np.isfinite(out).all());self.assertTrue(np.all((out>=0)&(out<=1)))
    def test_globes_recede_behind_lids_without_moving_measured_face(self):
        p=np.zeros((474,3));labels=np.zeros(len(p),np.uint8)
        # The sphere would otherwise extend two mm through the lids.
        p[468:]=[[-.02,0,-.02],[.02,0,-.02],[0,-.02,-.02],[0,.02,-.02],[0,0,-.04],[0,0,0]];labels[468:]=1
        for i in EYES['imageLeft']['ring']+EYES['imageLeft']['lids']:p[i]=[0,0,-.002]
        out,shifts=fit_eye_depth(p,labels)
        np.testing.assert_array_equal(out[:468],p[:468]);self.assertAlmostEqual(shifts['imageLeft'],2.6)
        self.assertLessEqual(out[468:,2].max(),-.0026+1e-9)
    def test_recorded_iris_survives_while_unobserved_sclera_is_estimated(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'capture.json').write_text('{}');Image.fromarray(self.photo).save(p/'crop.png')
            candidate={**self.candidate,'id':'left-frame','eye':'imageLeft','crop':'crop.png',
                       'trackedIris':{'center':[.5,.5],'radius':.2}}
            with patch('scripts.eye_detail.candidates',return_value=[candidate]),patch('scripts.eye_detail.request') as request:
                result=scan_eyes(p,[],False);request.assert_not_called()
            self.assertEqual(result['eyes']['imageLeft']['mode'],'recorded-iris')
            self.assertEqual(result['eyes']['imageRight']['mode'],'procedural-fallback')
            self.assertIn('1/2',result['summary']);self.assertTrue(result['geometryEstimated'])
    def test_api_failure_is_not_reported_as_astra_or_measured(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'capture.json').write_text('{}')
            with patch('scripts.eye_detail.candidates',return_value=[]),patch('scripts.eye_detail.request',side_effect=ValueError('Unavailable')):
                result=scan_eyes(p,[],True)
            self.assertIsNone(result['model']);self.assertEqual(result['apiError'],'Unavailable')
            self.assertTrue(all(e['mode']=='procedural-fallback' for e in result['eyes'].values()))
            self.assertIn('unavailable',result['summary'])
    def test_successful_generated_spec_is_cached_and_clearly_attributed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'capture.json').write_text('{}')
            with patch('scripts.eye_detail.candidates',return_value=[]),patch('scripts.eye_detail.request',return_value={name:default_spec() for name in ['imageLeft','imageRight']}) as request:
                first=scan_eyes(p,[],True);second=scan_eyes(p,[],True)
                self.assertEqual(request.call_count,1)
            self.assertEqual(first,second);self.assertEqual(first['model'],'gpt-6-astra')
            self.assertTrue(all(e['mode']=='astra-generated' and not e['photoUsable'] for e in first['eyes'].values()))
            self.assertTrue(first['geometryEstimated'])
    def test_local_only_never_calls_astra(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'capture.json').write_text('{}')
            with patch('scripts.eye_detail.candidates',return_value=[]),patch('scripts.eye_detail.request') as request:
                result=scan_eyes(p,[],False);request.assert_not_called()
            self.assertIn('disabled',result['summary'])

if __name__=='__main__':unittest.main()
