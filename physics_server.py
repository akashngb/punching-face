"""Loopback-only Newton CPU service. No photo, API key or arbitrary path access."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import json,re,threading,time,uuid
import numpy as np
from newton_face import NewtonFace,load_cage
ROOT=Path(__file__).resolve().parent
LOCK=threading.RLock();SESSIONS={}

def finite(value,low,high):
    value=float(value)
    if not np.isfinite(value) or not low<=value<=high:raise ValueError('Physics parameter outside allowed range.')
    return value

def dispatch(path,data):
    with LOCK:
        now=time.monotonic()
        for key,item in list(SESSIONS.items()):
            if key!=data.get('session') and now-item['used']>300:del SESSIONS[key]
        if path=='/physics/open':
            identifier=data.get('id','')
            if not re.fullmatch('[a-f0-9]{32}',identifier):raise ValueError('Invalid scan identifier.')
            folder=ROOT/'.local/face-captures'/identifier
            state=json.loads((folder/'status.json').read_text())
            if not state.get('photoModel'):raise ValueError('Create the photo mesh before starting Newton.')
            client=data.get('client')
            if client is not None and (not isinstance(client,str) or not re.fullmatch('[a-f0-9-]{36}',client)):raise ValueError('Invalid physics client.')
            owned=[key for key,item in SESSIONS.items() if client is not None and item.get('client')==client]
            if len(SESSIONS)-len(owned)>=4:raise ValueError('Close an unused model before opening another physics session.')
            cage=load_cage(folder);sim=NewtonFace(cage,finite(data.get('softness',.6),0,1));session=uuid.uuid4().hex
            # Compile and check the rest state before reporting ready.
            check=sim.step(1/240)
            if check['peakMm']>.05 or check['minimumVolumeRatio']<.95:raise ValueError('Newton rest-state validation failed.')
            for key in owned:del SESSIONS[key]
            SESSIONS[session]={'sim':sim,'cage':cage,'used':now,'folder':folder,'client':client}
            return {'session':session,**sim.info()}
        session=data.get('session');item=SESSIONS.get(session)
        if not item:raise ValueError('Physics session expired. Reload the photo model.')
        if path=='/physics/close':del SESSIONS[session];return {'closed':True}
        if not item['folder'].exists():del SESSIONS[session];raise ValueError('The source scan was deleted.')
        item['used']=now
        if path=='/physics/reset' or data.get('pose') is not None:
            cage=dict(item['cage'])
            if data.get('pose') is not None:
                pose=np.asarray(data['pose'],float)
                if pose.shape!=(1404,) or not np.isfinite(pose).all() or np.max(np.abs(pose))>.5:raise ValueError('Invalid posed cage.')
                cage['positions']=pose.tolist()
            item['sim']=NewtonFace(cage,finite(data.get('softness',.6),0,1))
        sim=item['sim']
        if path=='/physics/reset':return {'reset':True,**sim.info()}
        if path!='/physics/step':raise ValueError('Unknown physics endpoint.')
        impacts=data.get('impacts',[])
        if not isinstance(impacts,list) or len(impacts)>2:raise ValueError('Too many contacts.')
        for hit in impacts:sim.impact(hit['point'],hit['direction'],finite(hit['speed'],0,4))
        return sim.step(finite(data.get('dt',1/30),1/1000,1/30))

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_POST(self):
        try:
            if self.headers.get('Host','').split(':')[0] not in ('127.0.0.1','localhost'):raise ValueError('Local clients only.')
            if self.headers.get('Origin') not in (None,'http://127.0.0.1:5173','http://localhost:5173'):raise ValueError('Invalid request origin.')
            size=int(self.headers.get('Content-Length',0))
            if not 0<size<=100000 or self.headers.get('Content-Type','').split(';')[0]!='application/json':raise ValueError('Expected a bounded JSON request.')
            result=dispatch(self.path,json.loads(self.rfile.read(size)));code=200
        except (ValueError,KeyError,TypeError,OSError) as e:result={'error':str(e)};code=400
        except Exception:result={'error':'Newton could not advance the simulation. Reload the model.'};code=500
        body=json.dumps(result,allow_nan=False).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass

if __name__=='__main__':
    print('Newton CPU service http://127.0.0.1:5175',flush=True)
    # Optional Sentry tracing (SPONSOR_SETUP.md). /physics/step is sampled at 2% there. A no-op without a DSN.
    try:import sponsor_obs;sponsor_obs.init('physics');sponsor_obs.instrument_http(Handler)
    except ImportError:pass
    ThreadingHTTPServer(('127.0.0.1',5175),Handler).serve_forever()
