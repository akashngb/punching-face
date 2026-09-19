"""Exercise the installed Newton engine against a locally reconstructed cage.
Run with the Newton environment and a capture folder; no photos are uploaded.
"""
from pathlib import Path
import sys,json,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from newton_face import NewtonFace,load_cage
folder=Path(sys.argv[1]);cage=load_cage(folder);results=[]
for landmark in [50,280,14,1]:
    sim=NewtonFace(cage)
    for _ in range(12):rest=sim.step()
    assert rest['peakMm']<.05,('unstable rest state',rest['peakMm'])
    assert rest['minimumVolumeRatio']>.98
    sim.impact(sim.rest[landmark],[.3,0,-1],2.)
    peak=0.;minimum=1.;most_affected=0
    for _ in range(120):
        result=sim.step();peak=max(peak,result['peakMm']);minimum=min(minimum,result['minimumVolumeRatio'])
        displacement=np.linalg.norm(np.array(result['offsets']).reshape(-1,3),axis=1)
        most_affected=max(most_affected,int(np.sum(displacement>.0005)))
    assert peak>.3,('contact did not deform the surface',landmark,peak)
    assert peak<15,('unbounded surface deformation',landmark,peak)
    assert minimum>.115,('inverted or collapsed volume',landmark,minimum)
    assert result['peakMm']<.15,('face did not recover',landmark,result['peakMm'])
    assert most_affected<468*.7,('impact became global rigid motion',landmark,most_affected)
    results.append({'landmark':landmark,'peakMm':peak,'finalMm':result['peakMm'],'minimumVolumeRatio':minimum,'affectedNodes':most_affected,'stepMs':result['stepMs']})
    print(json.dumps(results[-1]),flush=True)
print('PASS: Newton rest stability, localized contact, non-inversion and recovery for cheeks, lip and nose.',flush=True)
(folder/'newton-validation.json').write_text(json.dumps({'engine':sim.info(),'results':results},indent=2))

# Resuming a deliberately held impact must not expire its active session.
from unittest.mock import patch
import tempfile
import physics_server
class PausedSimulation:
    def step(self,dt):return {'resumed':True}
with tempfile.TemporaryDirectory() as d:
    physics_server.SESSIONS['held']={'sim':PausedSimulation(),'folder':Path(d),'used':0.}
    with patch.object(physics_server.time,'monotonic',return_value=900.):
        assert physics_server.dispatch('/physics/step',{'session':'held','dt':1/30})['resumed']
    physics_server.SESSIONS.clear()
print('PASS: a held Newton session resumes after a long inspection pause.')
