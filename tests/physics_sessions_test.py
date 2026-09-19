"""Reloads reclaim their own Newton slot without evicting another tab."""
import json,sys,tempfile,unittest,uuid
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import physics_server as service

class FakeSimulation:
    def __init__(self,*args):pass
    def step(self,*args):return {'peakMm':0,'minimumVolumeRatio':1}
    def info(self):return {'engine':'test'}

class SessionLeaseTest(unittest.TestCase):
    def test_reload_at_capacity_replaces_only_its_own_tab(self):
        with tempfile.TemporaryDirectory() as root:
            identifier='a'*32
            folder=Path(root)/'.local/face-captures'/identifier
            folder.mkdir(parents=True)
            (folder/'status.json').write_text(json.dumps({'photoModel':True}))
            with patch.object(service,'ROOT',Path(root)),patch.object(service,'NewtonFace',FakeSimulation),patch.object(service,'load_cage',return_value={}),patch.object(service,'SESSIONS',{}):
                clients=[str(uuid.uuid4()) for _ in range(4)]
                sessions=[service.dispatch('/physics/open',{'id':identifier,'client':c})['session'] for c in clients]
                replacement=service.dispatch('/physics/open',{'id':identifier,'client':clients[0]})['session']
                self.assertEqual(len(service.SESSIONS),4)
                self.assertNotIn(sessions[0],service.SESSIONS)
                self.assertIn(replacement,service.SESSIONS)
                self.assertTrue(all(s in service.SESSIONS for s in sessions[1:]))
                with self.assertRaisesRegex(ValueError,'Close an unused model'):
                    service.dispatch('/physics/open',{'id':identifier,'client':str(uuid.uuid4())})

if __name__=='__main__':unittest.main()
