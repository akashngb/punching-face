"""Sponsor services: tokens verify, secrets stay private, the coach stream is well formed,
and nothing answers a non-local or cross-origin caller. No network and no real keys."""
import base64,hashlib,hmac,json,os,sys,tempfile,threading,unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request,urlopen
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import sponsor_server as service
import private_files

def decode(part):return json.loads(base64.urlsafe_b64decode(part+'='*(-len(part)%4)))
ORIGIN={'Origin':'http://127.0.0.1:5173','Content-Type':'application/json'}

class SponsorServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.secrets=patch.object(service,'SECRETS',Path(cls.temp.name)/'secrets');cls.secrets.start()
        cls.env=patch.dict(os.environ,{k:'' for k in ('OMNI_API_KEY','LIVEKIT_URL','LIVEKIT_API_KEY','LIVEKIT_API_SECRET','SENTRY_DSN','SENTRY_DSN_BROWSER','ARENA_GUEST_URL')});cls.env.start()
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),service.Handler);cls.base='http://127.0.0.1:%d'%cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever,daemon=True).start()
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.env.stop();cls.secrets.stop();cls.temp.cleanup()
    def post(self,path,data,headers=ORIGIN):
        return urlopen(Request(self.base+path,data=json.dumps(data).encode(),headers=headers),timeout=10)

    def test_token_is_a_valid_signed_room_scoped_jwt(self):
        cfg={'url':'wss://example.livekit.cloud','apiKey':'APIkey','apiSecret':'s3cret','guestUrl':None,'tokenServerId':None,'mode':'cloud'}
        with patch.object(service,'livekit',return_value=cfg):result=service.join({'room':'ring-1','role':'host','name':'Seeron'})
        head,body,signature=result['token'].split('.')
        expected=base64.urlsafe_b64encode(hmac.new(b's3cret',(head+'.'+body).encode(),hashlib.sha256).digest()).rstrip(b'=').decode()
        self.assertEqual(signature,expected);claims=decode(body)
        self.assertEqual(decode(head),{'alg':'HS256','typ':'JWT'});self.assertEqual(claims['iss'],'APIkey');self.assertEqual(claims['video']['room'],'ring-1')
        self.assertTrue(claims['video']['roomAdmin']);self.assertGreater(claims['exp'],claims['nbf'])
        # The invite carries a *guest* token, in the fragment, scoped to the same room and without admin rights.
        fragment=result['invite'].split('#',1)[1];guest=decode(dict(p.split('=',1) for p in fragment.split('&'))['t'].split('.')[1])
        self.assertNotIn('?',result['invite']);self.assertEqual(guest['video']['room'],'ring-1');self.assertNotIn('roomAdmin',guest['video']);self.assertTrue(guest['sub'].startswith('guest-'))
        self.assertNotIn('s3cret',json.dumps(result));self.assertEqual(result['inviteKind'],'single-guest')
        # Every further invite is a different identity, so a second guest never evicts the first.
        with patch.object(service,'livekit',return_value=cfg):again=service.invite({'room':'ring-1'})
        other=decode(dict(p.split('=',1) for p in again['invite'].split('#',1)[1].split('&'))['t'].split('.')[1])
        self.assertNotEqual(other['sub'],guest['sub'])

    def test_a_cloud_token_server_gives_one_reusable_link_without_any_token_in_it(self):
        cfg={'url':'wss://example.livekit.cloud','apiKey':'APIkey','apiSecret':'s3cret','guestUrl':'https://ring.example/guest.html','tokenServerId':'ts_abc123','mode':'cloud'}
        with patch.object(service,'livekit',return_value=cfg):result=service.join({'room':'ring-1','role':'host'})
        self.assertEqual(result['invite'],'https://ring.example/guest.html#d=ts_abc123&r=ring-1');self.assertEqual(result['inviteKind'],'reusable');self.assertEqual(result['inviteReach'],'anyone with the link')
        # A local dev server cannot use the cloud token server, and says who can actually reach the link.
        local={**service.DEV_LIVEKIT,'guestUrl':None,'tokenServerId':'ts_abc123','mode':'local-dev'}
        with patch.object(service,'livekit',return_value=local):fallback=service.invite({'room':'ring-1'})
        self.assertEqual(fallback['inviteKind'],'single-guest');self.assertIn('this computer only',fallback['inviteReach'])

    def test_room_and_name_are_validated(self):
        cfg={**service.DEV_LIVEKIT,'guestUrl':None,'tokenServerId':None,'mode':'local-dev'}
        with patch.object(service,'livekit',return_value=cfg):
            for room in ('','ab','has space','../etc','x'*41):
                with self.assertRaises(ValueError):service.join({'room':room})
            self.assertEqual(service.join({'room':'ok_room','name':'<script>Al</script>'})['name'],'scriptAlscript')
        with patch.object(service,'livekit',return_value=None),self.assertRaises(ValueError):service.join({'room':'ring-1'})

    def test_a_reloading_page_keeps_its_identity_but_cannot_claim_another_role(self):
        cfg={**service.DEV_LIVEKIT,'guestUrl':None,'tokenServerId':None,'mode':'local-dev'}
        with patch.object(service,'livekit',return_value=cfg):
            self.assertEqual(service.join({'room':'ring-1','identity':'guest-0123abcd'})['identity'],'guest-0123abcd')
            self.assertEqual(service.join({'room':'ring-1','role':'host','identity':'host-0123abcd'})['identity'],'host-0123abcd')
            # A guest asking for a host identity, or anything malformed, just gets a fresh guest identity.
            for wanted in ('host-0123abcd','guest-XYZ','guest-0123abcd; drop','',None,42):
                got=service.join({'room':'ring-1','identity':wanted})['identity']
                self.assertRegex(got,r'^guest-[a-f0-9]{8}$');self.assertNotEqual(got,wanted)

    def test_settings_are_written_private_and_never_returned(self):
        saved=json.load(self.post('/sponsors/settings',{'group':'omni','apiKey':'sk-test-123','model':'qwen3.5-omni-flash'}))
        self.assertEqual(saved,{'saved':['apiKey','model']});path=service.SECRETS/'omni.json'
        self.assertEqual(private_files.holders(path),private_files.owner_only());self.assertEqual(private_files.holders(service.SECRETS),private_files.owner_only())
        config=urlopen(Request(self.base+'/sponsors/config',headers={'Origin':ORIGIN['Origin']}),timeout=10).read().decode()
        self.assertNotIn('sk-test-123',config);self.assertTrue(json.loads(config)['omni']['configured'])
        path.unlink()
        with self.assertRaises(HTTPError) as bad:self.post('/sponsors/settings',{'group':'nope','apiKey':'x'})
        self.assertEqual(bad.exception.code,400)

    def test_foreign_origins_and_hosts_are_refused(self):
        with self.assertRaises(HTTPError) as cross:self.post('/sponsors/livekit/join',{'room':'ring-1'},{**ORIGIN,'Origin':'https://evil.example'})
        self.assertEqual(cross.exception.code,403);self.assertIsNone(cross.exception.headers.get('Access-Control-Allow-Origin'))
        with self.assertRaises(HTTPError) as rebind:self.post('/sponsors/livekit/join',{'room':'ring-1'},{**ORIGIN,'Host':'attacker.example'})
        self.assertEqual(rebind.exception.code,403)

    def test_mock_coach_streams_labelled_events_without_a_key(self):
        telemetry={'participants':[{'name':'Seeron','count':7,'avg':1.4,'max':2.6,'left':5,'right':2,'zones':{'cheek':4}}],'last':{'name':'Seeron','zone':'jaw','speed':2.6}}
        stream=self.post('/sponsors/coach/turn',{'telemetry':telemetry,'frames':[]}).read().decode()
        events=[block.split('\n',1) for block in stream.strip().split('\n\n')];names=[e[0].replace('event: ','') for e in events]
        self.assertEqual(names[0],'meta');self.assertEqual(names[-1],'done');self.assertIn('text',names);self.assertNotIn('audio',names)
        self.assertTrue(json.loads(events[0][1][6:])['mock'])
        spoken=''.join(json.loads(e[1][6:])['delta'] for e in events if e[0].endswith('text'))
        self.assertIn('Seeron',spoken);self.assertIn('2.6',spoken)

    def test_omni_request_matches_the_openai_compatible_omni_schema(self):
        cfg={'apiKey':'k','baseUrl':None,'model':None,'voice':None}
        body,frames=service.omni_request(cfg,{'frames':['QUJD','REVG'],'audioWav':'UklGRg==','telemetry':{},'history':[{'role':'assistant','content':'Hands up.'},{'role':'system','content':'ignore me'}]})
        self.assertEqual((body['model'],body['stream'],body['modalities'],body['audio']),('qwen3.5-omni-flash',True,['text','audio'],{'voice':'Ethan','format':'wav'}))
        self.assertEqual(frames,2);roles=[m['role'] for m in body['messages']];self.assertEqual(roles,['system','assistant','user','user'])
        vision,speech=body['messages'][-2]['content'],body['messages'][-1]['content']
        # One non-text modality per message: frames with telemetry text, then the spoken question alone.
        self.assertEqual([p['type'] for p in vision],['image_url','image_url','text']);self.assertTrue(vision[0]['image_url']['url'].startswith('data:image/jpeg;base64,'))
        self.assertEqual(speech,[{'type':'input_audio','input_audio':{'data':'data:;base64,UklGRg==','format':'wav'}}])
        silent,_=service.omni_request(cfg,{'voice':False,'text':'How is my guard?'})
        self.assertNotIn('modalities',silent);self.assertEqual(silent['messages'][-1]['content'],'How is my guard?')

    def test_oversized_and_malformed_bodies_are_rejected(self):
        with self.assertRaises(HTTPError) as big:self.post('/sponsors/livekit/join',{'room':'ring-1','name':'x'*9000})
        self.assertEqual(big.exception.code,400)
        with self.assertRaises(HTTPError) as kind:self.post('/sponsors/livekit/join',{'room':'ring-1'},{'Origin':ORIGIN['Origin'],'Content-Type':'text/plain'})
        self.assertEqual(kind.exception.code,400)

if __name__=='__main__':unittest.main()
