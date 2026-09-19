"""Sentry wiring: one browser click reads as ONE trace across the HTTP service, the pipeline
subprocess and its PipelineTimer stages; nothing sensitive is attached; and everything is a no-op
without a DSN. Envelopes are captured in memory, so there is no network and no real project."""
import json,os,subprocess,sys,tempfile,threading,unittest,unittest.mock
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request,urlopen
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import sentry_sdk
from sentry_sdk.transport import Transport
import sponsor_obs

class Memory(Transport):
    items=[]
    def capture_envelope(self,envelope):
        for item in envelope.items:Memory.items.append((item.headers.get('type'),item.payload.json if item.payload.json is not None else item.payload.bytes))
    def flush(self,timeout,callback=None):pass
    def kill(self):pass

def transactions():return [payload for kind,payload in Memory.items if kind=='transaction']
def streamed(kind):
    """AI spans and structured logs travel as their own batched envelope items (format v2), not inside the transaction."""
    sentry_sdk.flush();found=[]
    for item_kind,payload in Memory.items:
        if item_kind==kind:found+=(json.loads(payload) if isinstance(payload,(bytes,str)) else payload)['items']
    return found
TRACE,PARENT='0af7651916cd43dd8448eb211c80319c','b7ad6b7169203331'
# The physics interpreter (.local/newton-env) never runs the photo pipeline and lacks its dependencies.
try:import pipeline_timing;PIPELINE=None
except ImportError as missing:PIPELINE='the photo pipeline is not importable here (%s); run this file with .venv/bin/python too'%missing

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length',0)));Handler.child=sponsor_obs.child_env()
        body=b'{"ok":true}';self.send_response(202);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)

class Observability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.pop('SENTRY_DSN',None);assert sponsor_obs.init('test',transport=Memory)
        sponsor_obs.instrument_http(Handler);cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        threading.Thread(target=cls.server.serve_forever,daemon=True).start()
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close()

    def test_1_a_request_continues_the_browsers_trace_and_carries_no_body(self):
        secret='data:image/png;base64,FACEPIXELS'
        request=Request('http://127.0.0.1:%d/api/face-train?id=abc'%self.server.server_address[1],data=json.dumps({'image':secret}).encode(),headers={'Content-Type':'application/json','sentry-trace':TRACE+'-'+PARENT+'-1'})
        self.assertEqual(urlopen(request,timeout=10).status,202);sentry_sdk.flush()
        sent=transactions();self.assertEqual(len(sent),1);event=sent[0];trace=event['contexts']['trace']
        self.assertEqual((trace['trace_id'],trace['parent_span_id'],trace['op']),(TRACE,PARENT,'http.server'))
        self.assertEqual(event['transaction'],'POST /api/face-train','the query string (capture ids) stays out of the name')
        self.assertEqual(trace.get('status'),'ok');self.assertEqual(event['tags']['service'],'test','each process is tagged with the service name it passed to init()')
        self.assertNotIn('FACEPIXELS',json.dumps(event));self.assertNotIn('request',event)
        # The subprocess environment carries the SAME trace, so the pipeline joins it instead of starting its own.
        self.assertTrue(Handler.child['SENTRY_TRACE'].startswith(TRACE+'-'))

    @unittest.skipIf(PIPELINE,PIPELINE)
    def test_2_the_pipeline_subprocess_joins_that_trace_and_each_stage_is_a_span(self):
        from pipeline_timing import PipelineTimer
        sponsor_obs.patch_pipeline_timer();Memory.items.clear()
        with tempfile.TemporaryDirectory() as temp,unittest.mock.patch.dict(os.environ,{'SENTRY_TRACE':Handler.child['SENTRY_TRACE'],'SENTRY_BAGGAGE':Handler.child.get('SENTRY_BAGGAGE','')}):
            folder=Path(temp)/'0123abcd';folder.mkdir()
            with sponsor_obs.continue_from_env('build_photo_face'):
                timer=PipelineTimer(folder)
                for stage in ('cameras','surface','texture'):timer.mark(stage)
                timer.finish()
            saved=json.loads((folder/'timing.json').read_text())
        event=transactions()[0];self.assertEqual(event['contexts']['trace']['trace_id'],TRACE);self.assertEqual(event['transaction'],'build_photo_face')
        # Every stage also leaves a structured log that carries the trace id, so a log line opens its trace in Sentry.
        logs=[l for l in streamed('log') if l['body'].startswith('stage: ')]
        self.assertEqual([l['body'] for l in logs],['stage: cameras','stage: surface','stage: texture']);self.assertTrue(all(l['trace_id']==TRACE for l in logs))
        self.assertEqual(logs[0]['attributes']['capture']['value'],'0123abcd')
        spans=[(s['op'],s['description'],s.get('status')) for s in event['spans']]
        self.assertEqual(spans,[('pipeline.stage','cameras','ok'),('pipeline.stage','surface','ok'),('pipeline.stage','texture','ok')])
        # Wrapping must not change what the timer itself records.
        self.assertEqual([s['stage'] for s in saved['stages']],['cameras','surface','texture']);self.assertEqual(saved['status'],'complete')

    @unittest.skipIf(PIPELINE,PIPELINE)
    def test_3_a_failed_stage_is_marked_failed(self):
        from pipeline_timing import PipelineTimer
        Memory.items.clear()
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'feedbeef';folder.mkdir()
            with self.assertRaises(ValueError),sponsor_obs.continue_from_env('build_photo_face'):
                timer=PipelineTimer(folder);timer.mark('cameras');timer.finish('failed');raise ValueError('Camera recovery failed.')
        event=transactions()[0];self.assertEqual(event['contexts']['trace']['status'],'internal_error');self.assertEqual(event['spans'][0]['status'],'internal_error')

    def test_4_the_30hz_physics_loop_is_sampled_thinly_and_everything_else_is_kept(self):
        self.assertEqual(sponsor_obs._sampler({'transaction_context':{'name':'POST /physics/step'}}),.02)
        for name in ('POST /physics/open','POST /api/face-train','build_photo_face','POST /sponsors/coach/turn'):self.assertEqual(sponsor_obs._sampler({'transaction_context':{'name':name}}),1.0)

    def test_5_coach_calls_are_shaped_for_ai_monitoring(self):
        Memory.items.clear()
        with sentry_sdk.start_transaction(op='http.server',name='POST /sponsors/coach/turn'):
            with sponsor_obs.ai_span('qwen3.5-omni-flash','yibuapi',messages_count=4,system_prompt_len=487,frames_attached=4,has_voice=True,temperature=0.7,max_tokens=140,audio_ms=1500) as span:
                sponsor_obs.ai_usage(span,{'prompt_tokens':900,'completion_tokens':40,'total_tokens':940},612,4,finish_reason='stop',model='qwen3.5-omni-flash')
        spans=[s for s in streamed('span') if s['attributes'].get('sentry.op',{}).get('value')=='gen_ai.chat']
        self.assertEqual(len(spans),1);span=spans[0];self.assertEqual(span['name'],'chat qwen3.5-omni-flash');self.assertEqual(span['trace_id'],transactions()[0]['contexts']['trace']['trace_id'])
        for key,value in {'gen_ai.operation.name':'chat','gen_ai.request.model':'qwen3.5-omni-flash','gen_ai.system':'yibuapi',
                          'gen_ai.usage.input_tokens':900,'gen_ai.usage.output_tokens':40,'gen_ai.usage.total_tokens':940,
                          'gen_ai.response.first_token_ms':612,'gen_ai.response.finish_reason':'stop',
                          'gen_ai.request.frames_attached':4,'gen_ai.request.messages_count':4,'gen_ai.request.has_voice':True,
                          'gen_ai.request.temperature':0.7,'gen_ai.request.max_tokens':140,'gen_ai.request.audio_ms':1500,
                          'gen_ai.request.system_prompt_len':487}.items():
            self.assertEqual(span['attributes'][key]['value'],value)
        # Cost computed from the table in sponsor_obs.AI_PRICES ($0.10/M input + $0.30/M output for omni-flash).
        self.assertAlmostEqual(span['attributes']['gen_ai.usage.cost_usd']['value'],round(900*0.10/1e6+40*0.30/1e6,6),places=6)

    def test_6_without_a_dsn_every_call_is_a_harmless_no_op(self):
        code="import sponsor_obs as o\nassert o.init('x') is False and o.ENABLED is False\no.capture(ValueError('x'));o.instrument_http(object);o.patch_pipeline_timer()\nassert 'SENTRY_TRACE' not in o.child_env()\nwith o.continue_from_env('x') as t:assert t is None\nwith o.ai_span('m','s') as s:o.ai_usage(s,None,1,0)\nprint('noop-ok')"
        env={k:v for k,v in os.environ.items() if not k.startswith('SENTRY')};env['HOME']=tempfile.gettempdir()
        with tempfile.TemporaryDirectory() as empty:
            # Run from a copy of the module in an empty folder, so no .local/secrets/sentry.json can switch it on.
            (Path(empty)/'sponsor_obs.py').write_text((ROOT/'sponsor_obs.py').read_text())
            result=subprocess.run([sys.executable,'-c',code],cwd=empty,env=env,capture_output=True,text=True,timeout=60)
        self.assertEqual(result.stdout.strip(),'noop-ok',result.stderr)

if __name__=='__main__':unittest.main()
