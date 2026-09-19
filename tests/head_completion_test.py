"""Regression tests for crown pinching and protected captured geometry."""
import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.head_material import missing_head_material
from scripts.astra_head_completion import apply_shape_prior

class CompletionTests(unittest.TestCase):
    def test_crown_does_not_have_an_angular_texture_pole(self):
        p=np.zeros((470,3));p[10,1]=.1;p[152,1]=-.1;p[469]=[.1,.2,-.2]
        angle=np.linspace(0,2*np.pi,100);x=np.c_[np.cos(angle)*1e-7,np.full(100,.19),-.06+np.sin(angle)*1e-7];n=np.tile([0,1,0],(100,1))
        # Rear reference has high spatial variation so angular wrapping would
        # visibly fail this continuity bound around the former pole.
        yy,xx=np.mgrid[:200,:200];rear=np.stack([.1+.1*np.sin(xx/7),.1+.08*np.cos(yy/9),np.full_like(xx,.1,dtype=float)],2)
        rgb,scalp=missing_head_material(x,n,p,{'hair':{'present':True}},np.array([.6,.4,.3]),rear)
        self.assertTrue(np.isfinite(rgb).all());self.assertLess(np.ptp(rgb,axis=0).max(),1e-4);self.assertTrue((scalp==1).all())
    def test_ai_shape_cannot_move_measured_face_or_neck_cut(self):
        p=np.zeros((480,3));p[:,2]=-.2;p[:,1]=.07;p[10,1]=.1;p[152,1]=-.1;p[479,1]=-.124
        spec={'model':'fixture','head':{'posteriorDepthScale':1.1,'posteriorWidthScale':1.06,'crownLiftMm':6,'occiputLiftMm':5}}
        q,info=apply_shape_prior(p,np.array([[468,469,470],[471,472,473]]),1,spec,{})
        np.testing.assert_array_equal(q[:471],p[:471]);np.testing.assert_array_equal(q[479],p[479]);self.assertGreater(info['maxDisplacementMm'],0)
        q,info=apply_shape_prior(p,np.array([[468,469,470]]),1,spec,{'completeOrbit':True});np.testing.assert_array_equal(q,p);self.assertFalse(info['applied'])
if __name__=='__main__':unittest.main()
