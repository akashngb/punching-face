"""Loopback-only sponsor services: OMNI Live coach relay, LiveKit room tokens, client config.

Keys stay on this side of the browser: environment variables first, else
.local/secrets/{omni,livekit,sentry}.json (mode 0600). Nothing here reads captures or
photographs. The page calls this origin directly with CORS, so vite.config.js is untouched.
Standard library only, so it runs in the existing Python 3.9 venv.
"""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote
import base64,hashlib,hmac,json,os,re,socket,sys,time,urllib.error,urllib.request,uuid
import sponsor_obs
from private_files import restrict

# yibuapi call ledger (Huawei OMNI Live challenge requires per-call recording).
# Import the sponsor's canonical writer from `.local/third_party/` without copying
# it into the repo. Fails soft if the package isn't extracted yet.
_ROOT=Path(__file__).resolve().parent
_LEDGER=_ROOT/'.local/usage/yibu_api_calls.jsonl'
_LEDGER.parent.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('YIBU_AUDIT_LOG',str(_LEDGER))
_YIBU_PKG=_ROOT/'.local/third_party/yibuapi-examples/yibuapi_examples_20260918_v01'
if _YIBU_PKG.is_dir() and str(_YIBU_PKG) not in sys.path: sys.path.insert(0,str(_YIBU_PKG))
try:from yibu_audit import append_audit_record as _yibu_audit
except Exception:_yibu_audit=None

ROOT=Path(__file__).resolve().parent;SECRETS=ROOT/'.local/secrets';PORT=5176
ORIGINS=('http://127.0.0.1:5173','http://localhost:5173')
# `livekit-server --dev` ships these public credentials; they only ever work against a local dev server.
DEV_LIVEKIT={'url':'ws://127.0.0.1:7880','apiKey':'devkey','apiSecret':'secret'}
FIELDS={'omni':('apiKey','baseUrl','model','voice'),'livekit':('url','apiKey','apiSecret','guestUrl','tokenServerId'),'sentry':('browserDsn','pythonDsn','environment')}

COACH="""You are Cornerman, a boxing coach watching one or more people spar against a 3D head on a laptop.
You receive webcam keyframes of the person throwing, a spoken question (if any), and exact punch telemetry
measured on the device. Trust the telemetry for numbers and use the frames for form: guard height, elbow flare,
stance, whether they reset after punching. Speak like a coach between rounds: one or two short sentences, concrete,
one correction at a time, use names when several people are in the room. Never invent numbers that are not in the
telemetry. If a frame shows nothing useful, say what you need to see. Safety comes first: if someone sounds winded,
dizzy or in pain, tell them to stop and rest. This is solo training against a virtual target; never encourage
hitting a person."""

def secret(kind):
    """Environment wins, then the 0600 file. Returns only known fields."""
    env={'omni':{'apiKey':'OMNI_API_KEY','baseUrl':'OMNI_BASE_URL','model':'OMNI_MODEL','voice':'OMNI_VOICE'},
         'livekit':{'url':'LIVEKIT_URL','apiKey':'LIVEKIT_API_KEY','apiSecret':'LIVEKIT_API_SECRET','guestUrl':'ARENA_GUEST_URL','tokenServerId':'LIVEKIT_TOKEN_SERVER_ID'},
         'sentry':{'browserDsn':'SENTRY_DSN_BROWSER','pythonDsn':'SENTRY_DSN','environment':'SENTRY_ENVIRONMENT'}}[kind]
    saved={}
    try:saved=json.loads((SECRETS/(kind+'.json')).read_text())
    except (OSError,ValueError):pass
    return {k:os.environ.get(env[k]) or saved.get(k) for k in FIELDS[kind]}

def save_secret(kind,data):
    if kind not in FIELDS:raise ValueError('Unknown settings group.')
    clean={}
    for key in FIELDS[kind]:
        value=data.get(key)
        if value in (None,''):continue
        if not isinstance(value,str) or len(value)>600 or not value.isprintable():raise ValueError('Settings must be short printable text.')
        clean[key]=value.strip()
    if not clean:raise ValueError('Nothing to save.')
    SECRETS.mkdir(parents=True,exist_ok=True);restrict(SECRETS);path=SECRETS/(kind+'.json')
    merged={}
    try:merged=json.loads(path.read_text())
    except (OSError,ValueError):pass
    merged.update(clean);tmp=path.with_suffix('.tmp')
    # Create the file already private; never widen permissions, even briefly.
    fd=os.open(str(tmp),os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,'w') as out:json.dump(merged,out)
    tmp.replace(path);restrict(path);return sorted(clean)

def local_livekit_up():
    try:
        with socket.create_connection(('127.0.0.1',7880),timeout=.25):return True
    except OSError:return False

def livekit():
    cfg=secret('livekit')
    if cfg['url'] and cfg['apiKey'] and cfg['apiSecret']:return {**cfg,'mode':'cloud' if cfg['url'].startswith('wss://') else 'self-hosted'}
    if local_livekit_up():return {**DEV_LIVEKIT,'guestUrl':cfg['guestUrl'],'tokenServerId':None,'mode':'local-dev'}
    return None

def b64url(raw):return base64.urlsafe_b64encode(raw).rstrip(b'=').decode()

def livekit_token(cfg,room,identity,name,host,ttl=4*3600):
    """A LiveKit access token is a plain HS256 JWT; minting it needs no SDK."""
    now=int(time.time());grant={'room':room,'roomJoin':True,'canPublish':True,'canSubscribe':True,'canPublishData':True}
    if host:grant.update(roomCreate=True,roomAdmin=True)
    claims={'iss':cfg['apiKey'],'sub':identity,'name':name,'nbf':now-10,'exp':now+ttl,'video':grant,'metadata':json.dumps({'role':'host' if host else 'guest'})}
    signing=b64url(json.dumps({'alg':'HS256','typ':'JWT'},separators=(',',':')).encode())+'.'+b64url(json.dumps(claims,separators=(',',':')).encode())
    return signing+'.'+b64url(hmac.new(cfg['apiSecret'].encode(),signing.encode(),hashlib.sha256).digest())

def clean_name(value,fallback):
    value=re.sub(r'[^\w .\'-]','',str(value or ''),flags=re.UNICODE).strip()[:24]
    return value or fallback

def room_name(data):
    room=str(data.get('room') or '')
    if not re.fullmatch(r'[A-Za-z0-9_-]{3,40}',room):raise ValueError('Room names use 3-40 letters, digits, - or _.')
    return room

def configured():
    cfg=livekit()
    if not cfg:raise ValueError('LiveKit is not configured. Add keys in Arena settings, or run: livekit-server --dev --bind 127.0.0.1')
    return cfg

def invite(data):
    """A link for guests. A LiveKit identity lives inside its token, so a shared token would let each
    new guest evict the last one. With a LiveKit Cloud development token server the link carries only
    that public id and every guest mints a unique token; otherwise each call mints one single-guest link.
    Tokens ride in the URL fragment, which browsers never send to any server."""
    cfg=configured();room=room_name(data);base=cfg.get('guestUrl') or ORIGINS[0]+'/guest.html'
    public=base.startswith('https://') and cfg['mode']=='cloud'
    reach='anyone with the link' if public else 'this computer only: other devices need LiveKit Cloud (wss) and an https guest page'
    token_server=cfg.get('tokenServerId')
    if token_server and cfg['mode']=='cloud':
        if not re.fullmatch(r'[A-Za-z0-9_-]{4,80}',token_server):raise ValueError('The development token server id looks malformed.')
        return {'invite':base+'#d='+token_server+'&r='+room,'inviteKind':'reusable','inviteReach':reach}
    guest=livekit_token(cfg,room,'guest-'+uuid.uuid4().hex[:8],'Guest',False)
    return {'invite':base+'#u='+quote(cfg['url'],safe='')+'&r='+room+'&t='+guest,'inviteKind':'single-guest','inviteReach':reach}

def join(data):
    cfg=configured();room=room_name(data)
    host=data.get('role')=='host';prefix='host-' if host else 'guest-';name=clean_name(data.get('name'),'Host' if host else 'Guest')
    # A page that reloads asks for the identity it had, so LiveKit replaces its stale connection at once
    # instead of leaving a ghost participant (and, for the host, a second 'model' track) for ~20 s.
    wanted=str(data.get('identity') or '');identity=wanted if re.fullmatch(prefix+r'[a-f0-9]{8}',wanted) else prefix+uuid.uuid4().hex[:8]
    result={'url':cfg['url'],'room':room,'identity':identity,'name':name,'mode':cfg['mode'],'token':livekit_token(cfg,room,identity,name,host)}
    if host:result.update(invite(data))
    return result

def telemetry_text(t):
    if not isinstance(t,dict):return 'No punch telemetry yet.'
    lines=['Punch telemetry measured on the device (metres per second, last 30 s):']
    for p in (t.get('participants') or [])[:6]:
        lines.append('- {name}: {count} punches, avg {avg:.1f} m/s, max {mx:.1f} m/s, left/right {l}/{r}, zones {zones}'.format(name=clean_name(p.get('name'),'someone'),count=int(p.get('count',0)),avg=float(p.get('avg',0)),mx=float(p.get('max',0)),l=int(p.get('left',0)),r=int(p.get('right',0)),zones=json.dumps(p.get('zones',{}))[:120]))
    last=t.get('last')
    if isinstance(last,dict):lines.append('Most recent: {who} hit the {zone} at {speed:.1f} m/s.'.format(who=clean_name(last.get('name'),'someone'),zone=str(last.get('zone','face'))[:20],speed=float(last.get('speed',0))))
    if t.get('guard'):lines.append('Guard estimate: '+str(t['guard'])[:80])
    if t.get('trigger'):lines.append('This turn was triggered by: '+str(t['trigger'])[:80])
    return '\n'.join(lines)

def omni_request(cfg,data):
    """OpenAI-compatible body for Qwen-Omni. Images and audio travel in separate user
    messages because Omni models accept one non-text modality per message."""
    frames=[f for f in (data.get('frames') or [])[:4] if isinstance(f,str) and len(f)<400_000]
    parts=[{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+f}} for f in frames]
    parts.append({'type':'text','text':telemetry_text(data.get('telemetry'))+('\nThe frames are the last few seconds, oldest first.' if frames else '\nVision is switched off for this turn.')})
    history=[m for m in (data.get('history') or [])[-6:] if isinstance(m,dict) and m.get('role') in ('user','assistant') and isinstance(m.get('content'),str) and len(m['content'])<600]
    messages=[{'role':'system','content':COACH},*history,{'role':'user','content':parts}]
    audio=data.get('audioWav');text=str(data.get('text') or '')[:400]
    if isinstance(audio,str) and 0<len(audio)<3_000_000:messages.append({'role':'user','content':[{'type':'input_audio','input_audio':{'data':'data:;base64,'+audio,'format':'wav'}}]})
    elif text:messages.append({'role':'user','content':text})
    else:messages.append({'role':'user','content':'Give me one coaching cue from what you just saw.'})
    body={'model':cfg['model'] or 'qwen3.5-omni-flash','messages':messages,'stream':True,'stream_options':{'include_usage':True},'max_tokens':140,'temperature':.7}
    if data.get('voice',True):body.update(modalities=['text','audio'],audio={'voice':cfg['voice'] or 'Ethan','format':'wav'})
    return body,len(frames)

def mock_reply(data):
    t=data.get('telemetry') or {};last=t.get('last') or {};people=t.get('participants') or []
    if not people:return 'I have not seen a punch yet. Hands up, chin down, and throw a jab when you are ready.'
    top=max(people,key=lambda p:p.get('max',0));side='left' if top.get('left',0)>top.get('right',0) else 'right'
    return '{name}, {count} punches, top speed {mx:.1f} metres per second. You favour the {side}; mix in the other hand and bring your guard back after the {zone}.'.format(name=clean_name(top.get('name'),'Fighter'),count=int(top.get('count',0)),mx=float(top.get('max',0)),side=side,zone=str(last.get('zone','cheek'))[:20])

class Handler(BaseHTTPRequestHandler):
    server_version='PunchingFaceSponsors/1';protocol_version='HTTP/1.1'
    def log_message(self,*args):pass
    def cors(self):
        origin=self.headers.get('Origin')
        if origin in ORIGINS:self.send_header('Access-Control-Allow-Origin',origin);self.send_header('Vary','Origin');self.send_header('Access-Control-Allow-Headers','Content-Type, sentry-trace, baggage');self.send_header('Access-Control-Allow-Methods','GET, POST, OPTIONS')
    def guard(self):
        if self.headers.get('Host','').split(':')[0] not in ('127.0.0.1','localhost'):raise PermissionError('Local clients only.')
        if self.headers.get('Origin') not in (None,*ORIGINS):raise PermissionError('Invalid request origin.')
    def reply(self,code,payload):
        body=json.dumps(payload,allow_nan=False).encode();self.send_response(code);self.cors();self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass
    def do_OPTIONS(self):
        self.send_response(204);self.cors();self.send_header('Content-Length','0');self.end_headers()
    def do_GET(self):
        try:
            self.guard()
            if self.path!='/sponsors/config':return self.reply(404,{'error':'Unknown sponsor endpoint.'})
            omni=secret('omni');lk=livekit();sentry=secret('sentry')
            # The Sentry DSN is public by design; no other secret ever leaves this process.
            self.reply(200,{'omni':{'configured':bool(omni['apiKey']),'model':omni['model'] or 'qwen3.5-omni-flash','voice':omni['voice'] or 'Ethan','gateway':(omni['baseUrl'] or 'https://yibuapi.com/v1').split('/')[2]},
                'livekit':{'configured':bool(lk),'mode':lk['mode'] if lk else None,'url':lk['url'] if lk else None,'guestUrl':(lk or {}).get('guestUrl'),'reusableInvites':bool((lk or {}).get('tokenServerId'))},
                'sentry':{'dsn':sentry['browserDsn'] or sentry['pythonDsn'],'environment':sentry['environment'] or 'hackathon','python':sponsor_obs.ENABLED}})
        except PermissionError as e:self.reply(403,{'error':str(e)})
    def do_POST(self):
        try:
            self.guard();size=int(self.headers.get('Content-Length',0));limit=6_000_000 if self.path=='/sponsors/coach/turn' else 8192
            if not 0<size<=limit or self.headers.get('Content-Type','').split(';')[0]!='application/json':raise ValueError('Expected a bounded JSON request.')
            data=json.loads(self.rfile.read(size))
            if not isinstance(data,dict):raise ValueError('Expected a JSON object.')
            if self.path=='/sponsors/livekit/join':return self.reply(200,join(data))
            if self.path=='/sponsors/livekit/invite':return self.reply(200,invite(data))
            if self.path=='/sponsors/settings':return self.reply(200,{'saved':save_secret(str(data.get('group')),data)})
            if self.path=='/sponsors/coach/turn':return self.coach(data)
            self.reply(404,{'error':'Unknown sponsor endpoint.'})
        except PermissionError as e:self.reply(403,{'error':str(e)})
        except (ValueError,KeyError,TypeError) as e:self.reply(400,{'error':str(e)})
        except Exception as e:
            sponsor_obs.capture(e);self.reply(500,{'error':'Sponsor service failed. Check its terminal output.'})
    def event(self,name,payload):
        self.wfile.write(('event: %s\ndata: %s\n\n'%(name,json.dumps(payload))).encode());self.wfile.flush()
    def coach(self,data):
        cfg=secret('omni');self.send_response(200);self.cors();self.send_header('Content-Type','text/event-stream');self.send_header('Cache-Control','no-store');self.send_header('Connection','close');self.end_headers();self.close_connection=True
        started=time.perf_counter()
        try:
            if not cfg['apiKey']:
                # Development stand-in so the capture/playback loop can be built before a key arrives.
                # It is labelled in the stream and in the UI, and it never claims to be the OMNI model.
                self.event('meta',{'mock':True,'model':'mock (no OMNI key configured)'})
                for word in mock_reply(data).split(' '):self.event('text',{'delta':word+' '});time.sleep(.03)
                return self.event('done',{'mock':True,'ms':round((time.perf_counter()-started)*1000)})
            body,frames=omni_request(cfg,data);model=body['model']
            # Non-PII shape data for Sentry AI monitoring. Never prompt content or images.
            shape=dict(messages_count=len(body['messages']),system_prompt_len=len(COACH),frames_attached=frames,has_voice=('audio' in body),temperature=body.get('temperature'),max_tokens=body.get('max_tokens'),audio_ms=0)
            audio=data.get('audioWav');shape['audio_ms']=int(len(audio)*3/4/48) if isinstance(audio,str) else 0  # rough wav bytes->ms
            active_span=None
            with sponsor_obs.ai_span(model,'yibuapi',**shape) as span:
                active_span=span
                request=urllib.request.Request((cfg['baseUrl'] or 'https://yibuapi.com/v1').rstrip('/')+'/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+cfg['apiKey'],'Accept':'text/event-stream'})
                self.event('meta',{'mock':False,'model':model,'frames':frames,'voice':'audio' in body})
                first=None;usage=None;finish=None
                with urllib.request.urlopen(request,timeout=60) as upstream:
                    for raw in upstream:
                        line=raw.decode('utf-8','replace').strip()
                        if not line.startswith('data:'):continue
                        chunk=line[5:].strip()
                        if chunk=='[DONE]':break
                        try:piece=json.loads(chunk)
                        except ValueError:continue
                        usage=piece.get('usage') or usage
                        for choice in piece.get('choices') or []:
                            finish=choice.get('finish_reason') or finish
                            delta=choice.get('delta') or {};audio=delta.get('audio') or {}
                            words=delta.get('content') if isinstance(delta.get('content'),str) else audio.get('transcript')
                            if words or audio.get('data'):first=first or time.perf_counter()
                            if words:self.event('text',{'delta':words})
                            if audio.get('data'):self.event('audio',{'pcm16':audio['data'],'rate':24000})
                latency=round(((first or time.perf_counter())-started)*1000)
                sponsor_obs.ai_usage(span,usage,latency,frames,finish_reason=finish,model=model)
                sponsor_obs.log('coach.turn',model=model,frames=frames,first_token_ms=latency,finish_reason=finish or 'unknown',tokens=(usage or {}).get('total_tokens'))
                # Yibuapi challenge audit ledger (required per §6 of the reporting guide).
                if _yibu_audit is not None:
                    try:_yibu_audit(model=model,api_key=cfg.get('apiKey') or '',endpoint=(cfg['baseUrl'] or 'https://yibuapi.com/v1').rstrip('/')+'/chat/completions',purpose=str(data.get('purpose') or 'punching-face.coach'),transport='http',ok=True,status_code=200,latency_s=time.perf_counter()-started,response_json={'usage':usage or {}})
                    except Exception:pass
                self.event('done',{'mock':False,'firstTokenMs':latency,'ms':round((time.perf_counter()-started)*1000),'usage':usage,'finishReason':finish})
        except urllib.error.HTTPError as e:
            detail=e.read(600).decode('utf-8','replace');sponsor_obs.capture(e,{'status':e.code,'detail':detail})
            # Also mark the AI span so AI-monitoring filters (error rate, top failing models) see it.
            try:sponsor_obs.ai_error(active_span if 'active_span' in dir() else None,e.code,detail[:40])
            except Exception:pass
            # Failed calls are still recorded (guide §6). Never quote the key back into `detail`.
            if _yibu_audit is not None:
                try:_yibu_audit(model=cfg.get('model') or 'qwen3.5-omni-flash',api_key=cfg.get('apiKey') or '',endpoint=(cfg['baseUrl'] or 'https://yibuapi.com/v1').rstrip('/')+'/chat/completions',purpose=str(data.get('purpose') or 'punching-face.coach'),transport='http',ok=False,status_code=e.code,latency_s=time.perf_counter()-started,error=('HTTP %d: %s'%(e.code,detail[:200])))
                except Exception:pass
            self.event('error',{'status':e.code,'message':'OMNI gateway refused the request (%d).'%e.code,'detail':detail})
        except (BrokenPipeError,ConnectionResetError):pass
        except Exception as e:
            sponsor_obs.capture(e)
            try:self.event('error',{'message':'Coach relay failed: '+type(e).__name__})
            except OSError:pass

if __name__=='__main__':
    sponsor_obs.init('sponsor-server');sponsor_obs.instrument_http(Handler)
    print('Sponsor services http://127.0.0.1:%d  (OMNI key: %s · LiveKit: %s · Sentry: %s)'%(PORT,'yes' if secret('omni')['apiKey'] else 'no, mock coach',(livekit() or {}).get('mode','not configured'),'on' if sponsor_obs.ENABLED else 'off'),flush=True)
    ThreadingHTTPServer(('127.0.0.1',PORT),Handler).serve_forever()
