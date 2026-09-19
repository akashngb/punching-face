"""Local face capture API with incremental persistence and cancellable workers."""
from pathlib import Path
import base64, io, json, math, os, re, shutil, signal, subprocess, sys, threading, time, uuid
from PIL import Image
from openai_capture import config, configure, test_connection
# Optional Sentry (SPONSOR_SETUP.md): the pipeline subprocess joins the trace of the request that started it.
try:from sponsor_obs import child_env as trace_env
except ImportError:trace_env=lambda:None
ROOT=Path(__file__).resolve().parent

def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,allow_nan=False));tmp.replace(path)

def interrupt_timing(folder):
    path=folder/'timing.json'
    if not path.exists():return
    timing=json.loads(path.read_text())
    if timing.get('status')!='running':return
    now=time.time();stage=timing.pop('activeStage',None);started=timing.pop('stageStartedAt',now)
    if stage:timing.setdefault('stages',[]).append({'stage':stage,'seconds':round(max(0,now-started),3)})
    timing.update(status='failed',reconstructionSeconds=round(max(0,now-timing.get('requestedAt',now)),3));atomic(path,timing)

def coverage(frames):
    yaw=[f['yaw'] for f in frames if f.get('landmarks') and isinstance(f.get('yaw'),(int,float))]
    return {'frames':len(frames),'landmarkViews':len(yaw),'headOnlyViews':len(frames)-len(yaw),'yawMin':min(yaw,default=0),'yawMax':max(yaw,default=0),'span':max(yaw,default=0)-min(yaw,default=0)}

class FaceStore:
    def __init__(self,root=ROOT/'.local/face-captures',gpu_lock=None):
        self.root=Path(root);self.lock=threading.RLock();self.gpu_lock=gpu_lock or threading.Lock();self.jobs={};self.deleting=set()
        if self.root.exists():
            for p in self.root.glob('*/status.json'):
                try:
                    if json.loads(p.read_text()).get('status')=='running':
                        interrupt_timing(p.parent);atomic(p,{'status':'failed','stage':'interrupted','message':'The server restarted. Saved frames remain; restart reconstruction.'})
                except (OSError,ValueError):pass
    def folder(self,identifier):
        if not isinstance(identifier,str) or not re.fullmatch(r'[a-f0-9]{32}',identifier):raise ValueError('Invalid face scan identifier.')
        if identifier in self.deleting:raise ValueError('This scan is being deleted.')
        folder=self.root/identifier
        if not folder.is_dir():raise ValueError('Face scan not found.')
        return folder
    def create(self,fov=None,capture_region="face"):
        if capture_region not in ("face","head"):raise ValueError("Unknown capture region.")
        if fov is not None and (not isinstance(fov,(int,float)) or not math.isfinite(fov) or not 10<=fov<=120):raise ValueError('Camera field of view must be 10–120 degrees, or blank.')
        with self.lock:
            folder=self.root/uuid.uuid4().hex
            for name in ('images','masks'):(folder/name).mkdir(parents=True,exist_ok=True)
            atomic(folder/'capture.json',{'format':'punching-face-capture','version':1,'frames':[],'captureRegion':capture_region,'horizontalFovDegrees':fov})
            atomic(folder/'status.json',{'status':'captured','stage':'capture','message':'Ready to receive face frames.'})
            return {'id':folder.name,'frames':0}
    def append(self,identifier,frames):
        if not isinstance(frames,list) or not 1<=len(frames)<=6:raise ValueError('Send one to six frames per batch.')
        with self.lock:
            head_capture=json.loads((self.folder(identifier)/'capture.json').read_text()).get('captureRegion')=='head'
        # Validate the whole batch before writing any image or capture metadata.
        decoded=[]
        for frame in frames:
            yaw=frame.get('yaw');landmarks=frame.get('landmarks')
            if landmarks is None:
                if not head_capture or yaw is not None:raise ValueError('Only head scans can save views without facial landmarks; their viewing angle must be unknown.')
            else:
                if not isinstance(yaw,(int,float)) or not math.isfinite(yaw) or abs(yaw)>90:raise ValueError('Invalid face viewing angle.')
                if not isinstance(landmarks,list) or len(landmarks)!=468:raise ValueError('Each tracked face frame needs 468 matched face landmarks.')
                if any(not isinstance(p,dict) or any(not isinstance(p.get(k),(int,float)) or not math.isfinite(p[k]) or not 0<=p[k]<=1 for k in ('x','y')) for p in landmarks):raise ValueError('Face landmarks must be finite image coordinates.')
            iris=frame.get('irisLandmarks')
            if iris is not None:
                if landmarks is None or not isinstance(iris,list) or len(iris)!=10 or any(not isinstance(p,dict) or any(not isinstance(p.get(k),(int,float)) or not math.isfinite(p[k]) or not 0<=p[k]<=1 for k in ('x','y')) for p in iris):raise ValueError('Iris landmarks must be ten finite image coordinates on a tracked face.')
            timestamp=frame.get('timeSeconds')
            if timestamp is not None and (not isinstance(timestamp,(int,float)) or not math.isfinite(timestamp) or not 0<=timestamp<=600):raise ValueError('Invalid video timestamp.')
            encoded=frame.get('image','')
            if not isinstance(encoded,str) or not encoded.startswith('data:image/png;base64,') or len(encoded)>8_000_000:raise ValueError('Expected a masked PNG smaller than 6 MB.')
            raw=base64.b64decode(encoded.split(',',1)[1],validate=True)
            with Image.open(io.BytesIO(raw)) as im:
                if im.format!='PNG' or im.width*im.height>2_000_000 or min(im.size)<64 or im.mode!='RGBA':raise ValueError('Use an RGBA PNG with a face mask, up to two megapixels.')
                im.load();alpha=im.getchannel('A');mask=alpha.point(lambda x:255 if x>200 else 0)
                if not mask.getbbox():raise ValueError('The face mask is empty.')
                # Discard hidden RGB too; no background pixels are retained.
                clean=Image.new('RGBA',im.size);clean.paste(im,mask=mask);buf=io.BytesIO();clean.save(buf,format='PNG');raw=buf.getvalue()
                m=io.BytesIO();mask.save(m,format='PNG');maskraw=m.getvalue()
            decoded.append((raw,maskraw,{'yaw':yaw,'landmarks':landmarks,'irisLandmarks':iris,'viewKind':'face' if landmarks else 'head-only','timeSeconds':timestamp},im.size))
        with self.lock:
            folder=self.folder(identifier)
            if identifier in self.jobs:raise ValueError('Stop reconstruction before changing this scan.')
            manifest=json.loads((folder/'capture.json').read_text());previous=len(manifest['frames'])
            if previous+len(decoded)>240:raise ValueError('This scan already has the maximum 240 views.')
            sizes={item[3] for item in decoded}
            if len(sizes)!=1 or (manifest.get('imageSize') and tuple(manifest['imageSize']) not in sizes):raise ValueError('All views must use the same image dimensions and lens. Do not mix cropped or zoomed images.')
            manifest['imageSize']=list(decoded[0][3])
            written=[]
            try:
                for i,(raw,mask,metadata,size) in enumerate(decoded,previous):
                    name=f'frame_{i:04d}.png';metadata['filename']=name
                    for path,content in [(folder/'images'/name,raw),(folder/'masks'/(name+'.png'),mask)]:path.write_bytes(content);written.append(path)
                    manifest['frames'].append(metadata)
                atomic(folder/'capture.json',manifest)
            except Exception:
                for p in written:p.unlink(missing_ok=True)
                raise
            atomic(folder/'status.json',{'status':'captured','stage':'capture','message':f"{len(manifest['frames'])} face frames saved locally."})
            return {'id':identifier,**coverage(manifest['frames'])}
    def list(self):
        result=[]
        with self.lock:
            for folder in self.root.iterdir() if self.root.exists() else []:
                try:
                    data=json.loads((folder/'capture.json').read_text());state=json.loads((folder/'status.json').read_text())
                    result.append({'id':folder.name,'testFixture':bool(data.get('testFixture')),**coverage(data['frames']),**state,'savedAt':(folder/'capture.json').stat().st_mtime})
                except (OSError,ValueError,KeyError):pass
        return sorted(result,key=lambda v:v['savedAt'],reverse=True)
    def timing(self,identifier,data=None):
        with self.lock:
            folder=self.folder(identifier)
            if data is not None:
                if identifier in self.jobs and data.get('kind')!='load':raise ValueError('Timing metadata cannot change during reconstruction.')
                if data.get('kind')=='video':
                    for key,maximum in [('durationSeconds',300),('extractionSeconds',3600)]:
                        v=data.get(key)
                        if not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=maximum:raise ValueError('Invalid video timing.')
                    name=data.get('filename')
                    if not isinstance(name,str) or not 1<=len(name)<=255:raise ValueError('Invalid video name.')
                    previous=json.loads((folder/'source.json').read_text()) if (folder/'source.json').exists() else {}
                    atomic(folder/'source.json',{**previous,'kind':'video','filename':Path(name).name,'durationSeconds':data['durationSeconds'],'extractionSeconds':data['extractionSeconds'],'extractionComplete':data.get('extractionComplete') is True})
                elif data.get('kind')=='load':
                    seconds=data.get('seconds')
                    if not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or not 0<=seconds<=600:raise ValueError('Invalid load timing.')
                    path=folder/'timing.json'
                    if path.exists():
                        timing=json.loads(path.read_text())
                        if timing.get('status')=='complete' and 'loadSeconds' not in timing:timing['loadSeconds']=round(seconds,3);atomic(path,timing)
                else:raise ValueError('Unknown timing metadata.')
            return {key:json.loads((folder/name).read_text()) if (folder/name).exists() else None for key,name in [('source','source.json'),('timing','timing.json')]}
    def video(self,handler,identifier):
        folder=self.folder(identifier);path=folder/'source-video'
        if handler.command=='POST':
            size=int(handler.headers.get('Content-Length','0'));mime=handler.headers.get('Content-Type','').split(';')[0]
            if not 0<size<=500*1024*1024 or mime not in ('video/mp4','video/quicktime','video/webm','video/x-m4v','video/ogg','application/octet-stream'):raise ValueError('Use a supported video smaller than 500 MB.')
            with self.lock:
                source=json.loads((folder/'source.json').read_text());tmp=folder/'source-video.tmp'
                try:
                    with tmp.open('wb') as output:
                        remaining=size
                        while remaining:
                            chunk=handler.rfile.read(min(1024*1024,remaining))
                            if not chunk:raise ValueError('Video upload was interrupted.')
                            output.write(chunk);remaining-=len(chunk)
                    tmp.replace(path);source.update(videoStored=True,videoContentType=mime);atomic(folder/'source.json',source)
                finally:tmp.unlink(missing_ok=True)
            return 201,{'saved':True}
        if not path.exists():raise ValueError('The original video is not saved for this scan.')
        source=json.loads((folder/'source.json').read_text());size=path.stat().st_size;start=0;end=size-1;requested=handler.headers.get('Range')
        if requested:
            match=re.fullmatch(r'bytes=(\d*)-(\d*)',requested)
            if not match or not any(match.groups()):raise ValueError('Invalid video range.')
            a,b=match.groups()
            if a:start=int(a);end=min(end,int(b)) if b else end
            else:start=max(0,size-int(b))
            if start>end:
                handler.send_response(416);handler.send_header('Content-Range',f'bytes */{size}');handler.send_header('Content-Length','0');handler.end_headers();return None
        handler.send_response(206 if requested else 200);handler.send_header('Content-Type',source.get('videoContentType','application/octet-stream'));handler.send_header('Accept-Ranges','bytes');handler.send_header('Cache-Control','no-store');handler.send_header('Content-Length',str(end-start+1))
        if requested:handler.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        handler.end_headers()
        with path.open('rb') as file:
            file.seek(start);remaining=end-start+1
            while remaining:
                chunk=file.read(min(1024*1024,remaining))
                if not chunk:break
                handler.wfile.write(chunk);remaining-=len(chunk)
        return None
    def train(self,identifier,cloud,refine=False):
        with self.lock:
            folder=self.folder(identifier);frames=json.loads((folder/'capture.json').read_text())['frames'];c=coverage(frames)
            if len(frames)<24:raise ValueError('Capture at least 24 sharp views; aim for 60–100.')
            if c['landmarkViews']<12:raise ValueError('At least 12 front and three-quarter views need visible facial landmarks. Start and finish facing the camera.')
            if c['yawMin']>-25 or c['yawMax']<25 or not any(f.get('landmarks') and abs(f['yaw'])<10 for f in frames):raise ValueError('Capture the front and both sides, at least 25 degrees in each direction.')
            if cloud and not config()[0]:raise ValueError('Configure OpenAI or turn off cloud review.')
            if not self.gpu_lock.acquire(blocking=False):raise ValueError('Another reconstruction is using the trainer. Wait for it to finish.')
            previous=json.loads((folder/'status.json').read_text())
            if refine and not (folder/'poisson-status.json').exists():atomic(folder/'poisson-status.json',previous)
            prior=json.loads((folder/'timing.json').read_text()) if (folder/'timing.json').exists() else {}
            attempts=prior.pop('previousAttempts',[])
            if prior.get('reconstructionSeconds') is not None:attempts.append(prior)
            atomic(folder/'timing.json',{'status':'running','requestedAt':time.time(),'stages':[],'previousAttempts':attempts})
            atomic(folder/'status.json',{'status':'running','stage':'queued','message':'Starting fitted surface reconstruction…' if refine else 'Starting face reconstruction…','evidence':previous.get('evidence',{}) if refine else {}})
            try:
                log=(folder/'pipeline.log').open('w')
                script='build_photo_face.py';flags=[] if cloud else ['--local-only']
                try:p=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script),str(folder)]+flags,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=trace_env())
                finally:log.close()
            except Exception:
                self.gpu_lock.release();interrupt_timing(folder);atomic(folder/'status.json',{'status':'failed','stage':'start','message':'Could not start reconstruction.'});raise
            done=threading.Event();self.jobs[identifier]=(p,done)
            threading.Thread(target=self._wait,args=(identifier,p,done),daemon=True).start()
            return {'id':identifier,'status':'running'}
    def _wait(self,identifier,p,done):
        try:
            try:p.wait(timeout=1800)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            with self.lock:
                folder=self.root/identifier
                if folder.exists():
                    state=json.loads((folder/'status.json').read_text())
                    if state.get('status')=='running':
                        interrupt_timing(folder);atomic(folder/'status.json',{'status':'failed','stage':'interrupted','message':'Reconstruction stopped before completion. Frames remain saved.'})
        finally:
            with self.lock:self.jobs.pop(identifier,None)
            self.gpu_lock.release();done.set()
    def delete(self,identifier):
        with self.lock:
            folder=self.folder(identifier);job=self.jobs.get(identifier);self.deleting.add(identifier)
            if job:
                try:os.killpg(job[0].pid,signal.SIGTERM)
                except ProcessLookupError:pass
        if job:
            if not job[1].wait(5):
                try:os.killpg(job[0].pid,signal.SIGKILL)
                except ProcessLookupError:pass
                if not job[1].wait(5):
                    with self.lock:self.deleting.discard(identifier)
                    raise ValueError('The worker is still stopping. Try Delete scan again.')
        with self.lock:
            try:shutil.rmtree(folder)
            finally:self.deleting.discard(identifier)
        return {'deleted':True,'id':identifier}
    def shutdown(self):
        # A development-server restart must not leave orphan trainers behind.
        with self.lock:jobs=list(self.jobs.values())
        for process,done in jobs:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        for process,done in jobs:
            if not done.wait(3):
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                done.wait(3)
    def route(self,handler,url):
        from urllib.parse import parse_qs
        path=url.path;identifier=parse_qs(url.query).get('id',[''])[0]
        if path=='/api/face-video':return self.video(handler,identifier)
        if handler.command=='GET':
            if path=='/api/openai-config':return 200,{'configured':bool(config()[0]),'model':config()[1]}
            if path=='/api/face-captures':return 200,{'captures':self.list()}
            if path=='/api/face-status':return 200,{**json.loads((self.folder(identifier)/'status.json').read_text()),**self.timing(identifier)}
            if path=='/api/face-asset':
                asset=parse_qs(url.query).get('asset',[''])[0]
                if asset not in ('mesh.json','face.ply','texture-atlas.json','appearance.png','appearance-roughness.png','physics-cage.json','physics-binding.json'):raise ValueError('Unknown face asset.')
                file=self.folder(identifier)/asset
                if not file.exists():raise ValueError('This asset is not available yet.')
                body=file.read_bytes();handler.send_response(200);handler.send_header('Content-Type',{'json':'application/json','png':'image/png','ply':'application/octet-stream'}[file.suffix[1:]]);handler.send_header('Cache-Control','no-store');handler.send_header('Content-Length',str(len(body)));handler.end_headers();handler.wfile.write(body)
                return None
        else:
            if handler.headers.get('Content-Type','').split(';')[0]!='application/json':raise ValueError('Expected JSON.')
            size=int(handler.headers.get('Content-Length','0'));limit=40_000_000 if path=='/api/face-frames' else 4096
            if not 0<size<=limit:raise ValueError('Request is empty or too large.')
            data=json.loads(handler.rfile.read(size));identifier=data.get('id','')
            if path=='/api/openai-config':configure(data.get('apiKey'),data.get('model','gpt-4o-mini'));return 200,{'configured':True,'model':config()[1]}
            if path=='/api/openai-test':return 200,test_connection()
            if path=='/api/face-captures':return 201,self.create(data.get('horizontalFovDegrees'),data.get('captureRegion','face'))
            if path=='/api/face-frames':return 201,self.append(identifier,data.get('frames'))
            if path=='/api/face-timing':return 200,self.timing(identifier,data)
            if path=='/api/face-train':return 202,self.train(identifier,data.get('cloudReview') is True)
            if path=='/api/face-refine':return 202,self.train(identifier,True,refine=True)
            if path=='/api/face-delete':return 200,self.delete(identifier)
        return 404,{'error':'Unknown face endpoint.'}
