"""Loopback reconstruction service; optional OpenAI review of selected face frames."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from pathlib import Path
import hashlib
import json
import threading
import subprocess
import tempfile
import sys
import os
import time
import socket
import base64
import uuid
import re
import io
import signal
import atexit
from PIL import Image
from scripts.collect_references import collect
from face_pipeline import FaceStore

ROOT = Path(__file__).parent

def _load_dotenv(path):
    if not path.exists():return
    for raw in path.read_text().splitlines():
        line=raw.strip()
        if not line or line.startswith('#') or '=' not in line:continue
        key,value=line.split('=',1)
        key=key.strip()
        value=value.strip().strip('"').strip("'")
        if key and key not in os.environ:os.environ[key]=value

_load_dotenv(ROOT/'.env')
CACHE = ROOT / ".local" / "conversions"
CACHE.mkdir(parents=True, exist_ok=True)
LOCK = threading.Lock()
ARM_LOCK = threading.Lock()
CAPTURES = ROOT / '.local' / 'arm-captures'
FACE_STORE = FaceStore(gpu_lock=ARM_LOCK)
FACE_ROUTES = {'/api/face-captures','/api/face-frames','/api/face-train','/api/face-status','/api/face-asset','/api/face-delete','/api/face-timing','/api/face-video','/api/openai-config','/api/openai-test'}

def run_arm(folder):
    try:
        with (folder / 'pipeline.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/train_arm.py'),str(folder)],stdout=log,stderr=subprocess.STDOUT,timeout=1500)
        state=json.loads((folder/'status.json').read_text())
        if state.get('status') not in ('complete','failed'):
            raise RuntimeError('The reconstruction process stopped before producing an arm. Captured photos and logs are retained.')
    except Exception as exc:
        (folder/'status.json').write_text(json.dumps({'status':'failed','message':str(exc)}))
    finally:
        ARM_LOCK.release()


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, data):
        body = json.dumps(data, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url=urlparse(self.path)
        if url.path in FACE_ROUTES:return self.face_request(url)
        if url.path=='/api/arm-drafts':
            drafts=[]
            for folder in CAPTURES.iterdir() if CAPTURES.exists() else []:
                try:
                    manifest=json.loads((folder/'capture.json').read_text())
                    if manifest.get('testFixture'):continue
                    state=json.loads((folder/'status.json').read_text())
                    drafts.append({'id':folder.name,'side':manifest['side'],'frames':len(manifest['frames']),'bins':len(set(f.get('orientationBin') for f in manifest['frames'])),'forearmCm':manifest.get('forearmCm'),'status':state['status'],'message':state.get('message'),'savedAt':(folder/'capture.json').stat().st_mtime})
                except (OSError,ValueError,KeyError):continue
            return self.reply(200,{'drafts':sorted(drafts,key=lambda d:d['savedAt'],reverse=True)})
        if url.path=='/api/saved-session':
            path=ROOT/'.local/exports/punching-face-session.json'
            if not path.exists():return self.reply(404,{'error':'No saved session.'})
            return self.reply(200,json.loads(path.read_text()))
        if url.path=='/api/references':
            path=ROOT/'.local/impact-references/catalog.json'
            return self.reply(200,json.loads(path.read_text()) if path.exists() else {'references':[]})
        if url.path=='/api/reference-image':
            identifier=parse_qs(url.query).get('id',[''])[0]
            if not re.fullmatch(r'user-[1-4]',identifier):return self.reply(404,{'error':'Reference not found.'})
            path=ROOT/'.local/impact-references'/f'{identifier}.png'
            if not path.exists():return self.reply(404,{'error':'Reference not found.'})
            body=path.read_bytes();self.send_response(200);self.send_header('Content-Type','image/png');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
        if url.path == '/api/arm-status':
            identifier=parse_qs(url.query).get('id',[''])[0]
            if not re.fullmatch(r'[a-f0-9]{32}',identifier):
                return self.reply(400,{'error':'Invalid capture identifier.'})
            path=CAPTURES/identifier/'status.json'
            if not path.exists():return self.reply(404,{'error':'Capture not found.'})
            try:return self.reply(200,json.loads(path.read_text()))
            except json.JSONDecodeError:return self.reply(200,{'status':'running','message':'Updating reconstruction status…'})
        if self.path == "/api/health":
            return self.reply(200, {"ok":True,"backend":"COLMAP / Brush / Open3D","cloud":"Optional OpenAI capture review"})
        self.reply(404, {"error":"Unknown endpoint"})

    def do_POST(self):
        url = urlparse(self.path)
        origin = self.headers.get("Origin", "")
        if origin and origin not in ("http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:5174"):
            return self.reply(403, {"error":"Only the local lab may request reconstruction."})
        if url.path in FACE_ROUTES:return self.face_request(url)
        if url.path=='/api/meshy-headshot':return self.meshy_headshot()
        if url.path=='/api/reference-search':
            return self.reply(200,collect())
        if url.path=='/api/reference-evidence':
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<500_000:raise ValueError('Evidence payload too large.')
                data=json.loads(self.rfile.read(size));identifier=data.get('referenceId','')
                if data.get('format')!='punching-face-impact-evidence' or not re.fullmatch(r'user-[1-4]',identifier):raise ValueError('Invalid reference evidence.')
                if len(data.get('landmarks',[]))!=468:raise ValueError('Expected 468 landmarks.')
                path=ROOT/'.local/impact-references'/f'{identifier}-landmarks.json';path.write_text(json.dumps(data,allow_nan=False))
                return self.reply(200,{'path':str(path)})
            except (ValueError,TypeError,OSError) as exc:return self.reply(422,{'error':str(exc)})
        if url.path == '/api/arm-train':
            if not ARM_LOCK.acquire(blocking=False):return self.reply(409,{'error':'An arm reconstruction is already running.'})
            launched=False
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=1000:raise ValueError('Invalid reconstruction request.')
                identifier=json.loads(self.rfile.read(size)).get('id','')
                if not re.fullmatch(r'[a-f0-9]{32}',identifier):raise ValueError('Invalid capture identifier.')
                folder=CAPTURES/identifier;manifest=json.loads((folder/'capture.json').read_text())
                if len(manifest['frames'])<12:raise ValueError('At least 12 captured views are needed to attempt reconstruction.')
                state=json.loads((folder/'status.json').read_text())
                if state.get('status')!='captured':raise ValueError('This capture already has a reconstruction attempt. Its images and results are retained.')
                (folder/'status.json').write_text(json.dumps({'status':'running','message':'Saved capture loaded. Starting camera reconstruction…'}))
                threading.Thread(target=run_arm,args=(folder,),daemon=True).start();launched=True
                return self.reply(202,{'id':identifier,'path':str(folder)})
            except (OSError,ValueError,KeyError,TypeError) as exc:return self.reply(422,{'error':str(exc)})
            finally:
                if not launched:ARM_LOCK.release()
        if url.path in ('/api/arm-capture','/api/arm-draft'):
            train=url.path=='/api/arm-capture'
            if train and not ARM_LOCK.acquire(blocking=False):return self.reply(409,{'error':'An arm reconstruction is already running.'})
            launched=False
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=240_000_000:raise ValueError('Capture must be smaller than 240 MB.')
                data=json.loads(self.rfile.read(size))
                if data.get('side') not in ('left','right') or not (12 if train else 1)<=len(data.get('frames',[]))<=100:raise ValueError('Select an arm and capture 1–100 views; reconstruction needs at least 12.')
                identifier=uuid.uuid4().hex;folder=CAPTURES/identifier
                for name in ('images','masks','originals'):(folder/name).mkdir(parents=True,exist_ok=True)
                for i,frame in enumerate(data['frames']):
                    if len(frame.get('pose',[]))!=33 or len(frame.get('hand',[]))!=21:raise ValueError('Capture is missing matched body and hand landmarks.')
                    filename=f'frame_{i:04d}.png';frame['filename']=filename
                    for key,directory in [('image','images'),('mask','masks'),('original','originals')]:
                        encoded=frame.pop(key)
                        if not encoded.startswith('data:image/png;base64,'):raise ValueError('Expected PNG capture frames.')
                        raw=base64.b64decode(encoded.split(',',1)[1],validate=True)
                        with Image.open(io.BytesIO(raw)) as im:
                            if im.width*im.height>4_000_000:raise ValueError('Capture resolution is too large.')
                            im.verify()
                        (folder/directory/(filename+'.png' if key=='mask' else filename)).write_bytes(raw)
                (folder/'capture.json').write_text(json.dumps(data,allow_nan=False))
                (folder/'status.json').write_text(json.dumps({'status':'running' if train else 'captured','message':'Capture saved. Starting camera reconstruction…' if train else 'Images saved locally. Ready for reconstruction when enough views are available.'}))
                if train:threading.Thread(target=run_arm,args=(folder,),daemon=True).start();launched=True
                return self.reply(202 if train else 201,{'id':identifier,'path':str(folder),'frames':len(data['frames'])})
            except (ValueError,KeyError,TypeError,OSError) as exc:return self.reply(422,{'error':str(exc)})
            finally:
                if train and not launched:ARM_LOCK.release()
        if url.path == "/api/save":
            try:
                kind = parse_qs(url.query).get("type", ["json"])[0]
                size = int(self.headers.get("Content-Length", "0"))
                # Full strand geometry plus portable color/roughness atlases
                # can exceed the old 60 MB portrait-only export ceiling.
                if kind not in ("json", "glb") or not 0 < size <= 180_000_000:
                    return self.reply(413, {"error":"Unsupported export or file too large."})
                body = self.rfile.read(size)
                if kind == "json":
                    parsed = json.loads(body)
                    if parsed.get("format") != "punching-face-session":
                        raise ValueError("Not a Punching Face session")
                elif body[:4] != b"glTF":
                    raise ValueError("Not a binary glTF")
                folder = ROOT / ".local" / "exports"
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / ("punching-face.glb" if kind == "glb" else "punching-face-session.json")
                path.write_bytes(body)
                return self.reply(200, {"path":str(path)})
            except (ValueError, OSError) as exc:
                return self.reply(422, {"error":str(exc)})
        if url.path != "/api/reconstruct":
            return self.reply(404, {"error":"Unknown endpoint"})
        if not LOCK.acquire(blocking=False):
            return self.reply(409, {"error":"A reconstruction is running. Wait for it to finish."})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 120_000_000:
                return self.reply(413, {"error":"Provide a nonempty cropped PLY/SPLAT smaller than 120 MB."})
            data = self.rfile.read(size)
            query = parse_qs(url.query)
            ext = query.get("format", ["ply"])[0]
            if ext not in ("ply", "splat"):
                raise ValueError("CPU conversion supports uncompressed PLY and 32-byte SPLAT files.")
            depth = min(8, max(6, int(query.get("depth", [8])[0])))
            key = hashlib.sha256(data + f"v3:{ext}:{depth}".encode()).hexdigest()
            path = CACHE / (key + ".json")
            if path.exists():
                result = json.loads(path.read_text())
                result["stats"]["cached"] = True
            else:
                # Open3D's native Poisson implementation may terminate the
                # process on pathological topology. Isolate it from the server.
                with tempfile.TemporaryDirectory(prefix="punching-face-") as temp:
                    incoming = Path(temp) / ("source." + ext)
                    outgoing = Path(temp) / "mesh.json"
                    incoming.write_bytes(data)
                    worker = subprocess.run([sys.executable,str(ROOT / "scripts/convert.py"),str(incoming),str(outgoing),"--depth",str(depth)],capture_output=True,text=True,timeout=90)
                    if worker.returncode != 0 or not outgoing.exists():
                        print(worker.stdout[-2000:],worker.stderr[-2000:],flush=True)
                        raise ValueError("Poisson could not build this surface. Check the crop and normals, or use a surface-aligned export.")
                    result = json.loads(outgoing.read_text())
                path.write_text(json.dumps(result, allow_nan=False))
            self.reply(200, result)
        except (ValueError, RuntimeError, KeyError, IndexError, subprocess.TimeoutExpired) as exc:
            self.reply(422, {"error":str(exc)})
        except Exception as exc:
            print(type(exc).__name__, str(exc), flush=True)
            self.reply(500, {"error":"Conversion failed. Check the server log and PLY format."})
        finally:
            LOCK.release()

    def meshy_headshot(self):
        key=os.environ.get('MESHY_API_KEY','').strip()
        if not key:return self.reply(422,{'error':'MESHY_API_KEY is not configured on the server.'})
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=20_000_000:return self.reply(413,{'error':'Upload a JPEG or PNG smaller than 20 MB.'})
            body=self.rfile.read(size)
        except (ValueError,OSError) as exc:return self.reply(422,{'error':str(exc)})
        if body[:8].startswith(b'\x89PNG\r\n\x1a\n'):mime='image/png'
        elif body[:3]==b'\xff\xd8\xff':mime='image/jpeg'
        else:return self.reply(422,{'error':'Send a JPEG or PNG photograph.'})
        data_uri=f'data:{mime};base64,'+base64.b64encode(body).decode()
        payload={'image_url':data_uri,'enable_pbr':True,'should_remesh':True,'should_texture':True,'ai_model':'latest','topology':'triangle','target_polycount':30000,'symmetry_mode':'auto'}
        auth={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
        try:
            create=Request('https://api.meshy.ai/openapi/v1/image-to-3d',data=json.dumps(payload).encode(),headers=auth)
            with urlopen(create,timeout=60) as r:task=json.load(r)
        except HTTPError as e:
            detail=e.read().decode('utf-8','ignore')[:400]
            print('meshy create HTTP',e.code,detail,flush=True)
            return self.reply(422,{'error':f'Meshy rejected the request (HTTP {e.code}). {detail}'})
        except (URLError,TimeoutError,socket.timeout):
            return self.reply(504,{'error':'Meshy could not be reached from this computer.'})
        task_id=task.get('result') or task.get('id')
        if not task_id:return self.reply(422,{'error':'Meshy did not return a task id.'})
        status_url=f'https://api.meshy.ai/openapi/v1/image-to-3d/{task_id}'
        deadline=time.time()+600
        state={}
        while time.time()<deadline:
            try:
                with urlopen(Request(status_url,headers={'Authorization':'Bearer '+key}),timeout=60) as r:state=json.load(r)
            except HTTPError as e:
                return self.reply(422,{'error':f'Meshy status HTTP {e.code}.'})
            except (URLError,TimeoutError,socket.timeout):
                time.sleep(4);continue
            status=state.get('status','')
            if status=='SUCCEEDED':break
            if status in ('FAILED','CANCELED','EXPIRED'):
                message=(state.get('task_error') or {}).get('message') or status.lower()
                return self.reply(422,{'error':f'Meshy task {status.lower()}: {message}'})
            time.sleep(4)
        else:
            return self.reply(504,{'error':'Meshy task timed out after 10 minutes.'})
        glb_url=(state.get('model_urls') or {}).get('glb')
        if not glb_url:return self.reply(422,{'error':'Meshy did not return a GLB model URL.'})
        try:
            with urlopen(Request(glb_url),timeout=180) as r:glb_bytes=r.read()
        except (HTTPError,URLError,TimeoutError,socket.timeout) as exc:
            return self.reply(502,{'error':'Meshy GLB download failed: '+str(exc)})
        if glb_bytes[:4]!=b'glTF':
            return self.reply(502,{'error':'Meshy returned an unexpected model payload.'})
        self.send_response(200)
        self.send_header('Content-Type','model/gltf-binary')
        self.send_header('Content-Length',str(len(glb_bytes)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Meshy-Task-Id',str(task_id))
        self.end_headers()
        self.wfile.write(glb_bytes)

    def face_request(self,url):
        if self.headers.get('Host','').split(':')[0] not in ('127.0.0.1','localhost'):
            return self.reply(403,{'error':'This service accepts local requests only.'})
        try:
            result=FACE_STORE.route(self,url)
            if result is not None:self.reply(*result)
        except (ValueError,KeyError,TypeError,OSError) as exc:
            message=str(exc) if isinstance(exc,ValueError) else 'The face request could not be completed.'
            self.reply(422,{'error':message})


if __name__ == "__main__":
    atexit.register(FACE_STORE.shutdown)
    def shutdown_server(signum,frame):
        FACE_STORE.shutdown()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM,shutdown_server)
    signal.signal(signal.SIGINT,shutdown_server)
    print("Local reconstruction: http://127.0.0.1:5174", flush=True)
    # Optional Sentry tracing (SPONSOR_SETUP.md): continues the browser's trace. A no-op without a DSN.
    try:import sponsor_obs;sponsor_obs.init('punching-face-api');sponsor_obs.instrument_http(Handler)
    except ImportError:pass
    ThreadingHTTPServer(("127.0.0.1", 5174), Handler).serve_forever()
