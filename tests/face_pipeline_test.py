import base64,io,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from face_pipeline import FaceStore
import openai_capture

def frame(yaw=0):
    im=Image.new('RGBA',(64,64),(130,90,70,255));im.putpixel((0,0),(255,123,45,0));b=io.BytesIO();im.save(b,format='PNG')
    return {'yaw':yaw,'landmarks':[{'x':.5,'y':.5} for _ in range(468)],'image':'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()}
class CaptureTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.store=FaceStore(Path(self.temp.name)/'scans');self.id=self.store.create()['id']
    def tearDown(self):self.temp.cleanup()
    def test_incremental_save_survives_restart_and_delete_removes_everything(self):
        self.store.append(self.id,[frame(-30),frame(0),frame(30)]);again=FaceStore(self.store.root);self.assertEqual(again.list()[0]['frames'],3)
        im=Image.open(self.store.folder(self.id)/'images/frame_0000.png');self.assertEqual(im.getpixel((0,0)),(0,0,0,0))
        self.store.delete(self.id);self.assertEqual(self.store.list(),[])
    def test_calibration_validation_and_mixed_dimensions_rejected(self):
        with self.assertRaises(ValueError):self.store.create(float('nan'))
        calibrated=self.store.create(60)
        self.assertEqual(json.loads((self.store.folder(calibrated['id'])/'capture.json').read_text())['horizontalFovDegrees'],60)
        self.store.append(self.id,[frame()]);different=frame();b=io.BytesIO();Image.new('RGBA',(128,64),(80,80,80,255)).save(b,format='PNG');different['image']='data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
        with self.assertRaisesRegex(ValueError,'same image dimensions'):self.store.append(self.id,[different])
        self.assertEqual(self.store.list()[-1]['frames'] if self.store.list()[-1]['id']==self.id else self.store.list()[0]['frames'],1)
    def test_bad_batch_is_atomic(self):
        bad=frame();bad['landmarks'][4]['x']=float('nan')
        with self.assertRaises(ValueError):self.store.append(self.id,[frame(),bad])
        self.assertEqual(self.store.list()[0]['frames'],0);self.assertEqual(list((self.store.folder(self.id)/'images').iterdir()),[])
    def test_optional_iris_measurements_survive_capture_and_are_validated(self):
        good=frame();good['irisLandmarks']=[{'x':.4,'y':.45} for _ in range(10)]
        self.store.append(self.id,[good]);saved=json.loads((self.store.folder(self.id)/'capture.json').read_text())['frames'][0]
        self.assertEqual(saved['irisLandmarks'],good['irisLandmarks'])
        bad=frame();bad['irisLandmarks']=[{'x':float('nan'),'y':.5}]*10
        with self.assertRaisesRegex(ValueError,'Iris landmarks'):self.store.append(self.id,[frame(),bad])
        self.assertEqual(self.store.list()[0]['frames'],1)
    def test_rear_frames_are_retained_without_inventing_face_landmarks(self):
        identifier=self.store.create(capture_region='head')['id'];rear=frame();rear.update(yaw=None,landmarks=None,timeSeconds=24.5)
        self.store.append(identifier,[frame(-30),rear,frame(30)])
        record=json.loads((self.store.folder(identifier)/'capture.json').read_text())['frames'][1]
        self.assertIsNone(record['landmarks']);self.assertIsNone(record['yaw']);self.assertEqual(record['viewKind'],'head-only')
        with self.assertRaises(ValueError):self.store.append(self.id,[rear])
        from face_pipeline import coverage
        self.assertEqual(coverage([rear])['landmarkViews'],0);self.assertEqual(coverage([rear])['span'],0)
    def test_capture_gate_rejects_single_photo_and_frontal_only(self):
        self.store.append(self.id,[frame()])
        with self.assertRaisesRegex(ValueError,'24'):self.store.train(self.id,False)
        for _ in range(4):self.store.append(self.id,[frame()]*6)
        with self.assertRaisesRegex(ValueError,'both sides'):self.store.train(self.id,False)
    def test_video_timing_persists_without_invalidating_capture_hash(self):
        from pipeline_timing import PipelineTimer
        path=self.store.folder(self.id);before=(path/'capture.json').read_bytes()
        self.store.timing(self.id,{'kind':'video','filename':'sample.mov','durationSeconds':24.3,'extractionSeconds':15.6,'extractionComplete':True})
        timer=PipelineTimer(path);timer.mark('cameras');timer.mark('texture');timer.finish()
        self.store.timing(self.id,{'kind':'load','seconds':1.25})
        self.store.timing(self.id,{'kind':'load','seconds':9})
        again=FaceStore(self.store.root).timing(self.id)
        self.assertEqual(again['source']['filename'],'sample.mov');self.assertEqual(again['timing']['loadSeconds'],1.25)
        self.assertEqual([s['stage'] for s in again['timing']['stages']],['cameras','texture'])
        self.assertEqual((path/'capture.json').read_bytes(),before)
        with self.assertRaises(ValueError):self.store.timing(self.id,{'kind':'load','seconds':float('nan')})
    def test_saved_video_supports_seeking_and_rejects_partial_uploads(self):
        self.store.timing(self.id,{'kind':'video','filename':'sample.mov','durationSeconds':24.3,'extractionSeconds':1,'extractionComplete':True})
        class Request:
            def __init__(self,method,headers,body=b''):
                self.command=method;self.headers=headers;self.rfile=io.BytesIO(body);self.wfile=io.BytesIO();self.response_headers={}
            def send_response(self,code):self.code=code
            def send_header(self,key,value):self.response_headers[key]=value
            def end_headers(self):pass
        self.store.video(Request('POST',{'Content-Type':'video/quicktime','Content-Length':'10'},b'0123456789'),self.id)
        self.assertTrue(self.store.timing(self.id)['source']['videoStored'])
        for value,expected in [('bytes=2-5',b'2345'),('bytes=-3',b'789'),('bytes=7-',b'789')]:
            request=Request('GET',{'Range':value});self.store.video(request,self.id);self.assertEqual(request.code,206);self.assertEqual(request.wfile.getvalue(),expected)
        request=Request('GET',{'Range':'bytes=20-'});self.store.video(request,self.id);self.assertEqual(request.code,416)
        with self.assertRaisesRegex(ValueError,'interrupted'):self.store.video(Request('POST',{'Content-Type':'video/mp4','Content-Length':'10'},b'123'),self.id)
        self.assertEqual((self.store.folder(self.id)/'source-video').read_bytes(),b'0123456789')
    def test_server_restart_closes_an_interrupted_stage_timer(self):
        import time
        from face_pipeline import atomic
        path=self.store.folder(self.id)
        atomic(path/'status.json',{'status':'running'})
        atomic(path/'timing.json',{'status':'running','requestedAt':time.time()-10,'activeStage':'cameras','stageStartedAt':time.time()-8,'stages':[]})
        timing=FaceStore(self.store.root).timing(self.id)['timing']
        self.assertEqual(timing['status'],'failed');self.assertNotIn('activeStage',timing);self.assertEqual(timing['stages'][0]['stage'],'cameras');self.assertGreaterEqual(timing['reconstructionSeconds'],10)
    def test_path_traversal_and_oversized_batch_rejected(self):
        with self.assertRaises(ValueError):self.store.folder('../secrets')
        with self.assertRaises(ValueError):self.store.append(self.id,[frame()]*7)
    def test_delete_stops_worker_before_removing_files(self):
        import subprocess,threading
        process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
        done=threading.Event();self.store.gpu_lock.acquire();self.store.jobs[self.id]=(process,done)
        threading.Thread(target=self.store._wait,args=(self.id,process,done),daemon=True).start()
        self.store.delete(self.id)
        self.assertIsNotNone(process.poll());self.assertTrue(done.is_set());self.assertFalse(self.store.gpu_lock.locked());self.assertFalse((self.store.root/self.id).exists())
    def test_api_config_never_returns_key(self):
        class Request:command='GET'
        from urllib.parse import urlparse
        with patch('face_pipeline.config',return_value=('test-secret','gpt-4o-mini')):
            code,result=self.store.route(Request(),urlparse('/api/openai-config'));self.assertEqual(code,200);self.assertNotIn('test-secret',json.dumps(result))
    def test_key_saved_with_private_permissions(self):
        with patch.object(openai_capture,'CONFIG',Path(self.temp.name)/'secrets/openai.json'):
            openai_capture.configure('sk-'+'x'*30);self.assertEqual(openai_capture.CONFIG.stat().st_mode&0o777,0o600)
    def test_provider_errors_do_not_echo_key(self):
        from urllib.error import HTTPError
        with patch.object(openai_capture,'config',return_value=('sk-secret','gpt-4o-mini')),patch.object(openai_capture,'urlopen',side_effect=HTTPError('https://api.openai.com/v1/responses',401,'sk-secret',{},None)):
            with self.assertRaisesRegex(ValueError,'rejected the key') as e:openai_capture.test_connection()
            self.assertNotIn('sk-secret',str(e.exception))
if __name__=='__main__':unittest.main()
