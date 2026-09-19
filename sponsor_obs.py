"""Optional Sentry wiring for the Python side. Every function is a no-op unless both
`sentry-sdk` is installed and a DSN is configured, so importing this can never break a service.

One browser click should read as ONE trace: page -> server.py -> pipeline subprocess -> each
PipelineTimer stage, plus the OMNI coach calls as AI spans. Photographs, landmarks and request
bodies are never attached; only names, durations, counts and status.
"""
from contextlib import contextmanager
from pathlib import Path
import json,os
ROOT=Path(__file__).resolve().parent;ENABLED=False;SERVICE='contact'
try:import sentry_sdk
except Exception:sentry_sdk=None
# The structured-log API is a submodule, not an attribute: probing with hasattr() would silently drop every log.
try:from sentry_sdk import logger as sentry_logs
except Exception:sentry_logs=None

def _dsn():
    saved={}
    try:saved=json.loads((ROOT/'.local/secrets/sentry.json').read_text())
    except (OSError,ValueError):pass
    return os.environ.get('SENTRY_DSN') or saved.get('pythonDsn') or saved.get('browserDsn'),os.environ.get('SENTRY_ENVIRONMENT') or saved.get('environment') or 'hackathon'

def _sampler(context):
    # /physics/step runs at 30 Hz. Keep a thin sample so its latency stays visible without
    # drowning the quota or adding per-frame overhead; everything else is rare and kept.
    name=(context.get('transaction_context') or {}).get('name','')
    return .02 if 'physics/step' in name else 1.0

def init(service,transport=None):
    """Call once per process. `transport` lets tests capture envelopes without a network."""
    global ENABLED,SERVICE
    SERVICE=service;dsn,environment=_dsn()
    if not sentry_sdk or not (dsn or transport):return False
    options=dict(dsn=dsn or 'https://public@o0.ingest.sentry.io/0',environment=environment,traces_sampler=_sampler,send_default_pii=False,max_request_body_size='never',include_local_variables=False,server_name='contact-'+service)
    if transport:options['transport']=transport
    try:sentry_sdk.init(enable_logs=True,profile_session_sample_rate=1.0,profile_lifecycle='trace',**options)
    except TypeError:sentry_sdk.init(**options)  # older sentry-sdk without logs/continuous profiling
    sentry_sdk.set_tag('service',service);ENABLED=True;return True

def capture(error,extra=None):
    if not ENABLED:return
    with sentry_sdk.new_scope() as scope:
        for key,value in (extra or {}).items():scope.set_extra(key,value)
        sentry_sdk.capture_exception(error)

def log(message,**attributes):
    """Structured Sentry log plus the usual stdout line that pipeline.log already collects."""
    print(message,flush=True)
    if ENABLED and sentry_logs:sentry_logs.info(message,attributes={'service':SERVICE,**attributes})

def instrument_http(handler):
    """Wrap a BaseHTTPRequestHandler so each request continues the browser's trace. The service name
    comes from init(): one process, one service tag."""
    if not ENABLED:return
    send=handler.send_response
    def send_response(self,code,message=None):
        self._sentry_status=code;return send(self,code,message)
    handler.send_response=send_response
    for method in ('do_GET','do_POST'):
        original=getattr(handler,method,None)
        if not original:continue
        def wrapped(self,original=original):
            route=self.path.split('?')[0]
            with sentry_sdk.isolation_scope():
                transaction=sentry_sdk.continue_trace(dict(self.headers.items()),op='http.server',name='%s %s'%(self.command,route),source='route')
                with sentry_sdk.start_transaction(transaction) as active:
                    try:return original(self)
                    finally:
                        status=getattr(self,'_sentry_status',None)
                        if status:active.set_http_status(status)
        setattr(handler,method,wrapped)

def child_env():
    """Environment for a subprocess that should join the current trace."""
    env=dict(os.environ)
    if ENABLED:
        trace,baggage=sentry_sdk.get_traceparent(),sentry_sdk.get_baggage()
        if trace:env['SENTRY_TRACE']=trace
        if baggage:env['SENTRY_BAGGAGE']=baggage
    return env

@contextmanager
def continue_from_env(name,op='pipeline'):
    """Transaction for a subprocess, parented to whatever request spawned it."""
    if not ENABLED:
        yield None;return
    headers={'sentry-trace':os.environ.get('SENTRY_TRACE',''),'baggage':os.environ.get('SENTRY_BAGGAGE','')}
    transaction=sentry_sdk.continue_trace(headers,op=op,name=name,source='task')
    with sentry_sdk.start_transaction(transaction) as active:
        try:yield active
        except BaseException:
            active.set_status('internal_error');raise
        finally:sentry_sdk.flush(timeout=4)

def patch_pipeline_timer():
    """Each PipelineTimer stage becomes a span, so Sentry shows the same stages as timing.json."""
    if not ENABLED:return
    from pipeline_timing import PipelineTimer
    mark,finish=PipelineTimer.mark,PipelineTimer.finish
    def close(timer,status='ok'):
        span=getattr(timer,'_sentry_span',None)
        if span:span.set_status(status);span.finish();timer._sentry_span=None
    def traced_mark(self,stage):
        close(self);self._sentry_span=sentry_sdk.start_span(op='pipeline.stage',name=str(stage));log('stage: '+str(stage),stage=str(stage),capture=self.folder.name)
        return mark(self,stage)
    def traced_finish(self,state='complete'):
        close(self,'ok' if state=='complete' else 'internal_error');result=finish(self,state)
        log('pipeline '+state,state=state,capture=self.folder.name,seconds=result.get('reconstructionSeconds'))
        return result
    PipelineTimer.mark,PipelineTimer.finish=traced_mark,traced_finish

@contextmanager
def ai_span(model,system):
    """Shaped for Sentry's AI agent monitoring (gen_ai.* attributes)."""
    if not ENABLED:
        yield None;return
    with sentry_sdk.start_span(op='gen_ai.chat',name='chat '+model) as span:
        span.set_data('gen_ai.operation.name','chat');span.set_data('gen_ai.system',system);span.set_data('gen_ai.request.model',model)
        yield span

def ai_usage(span,usage,first_token_ms,frames):
    if not span:return
    usage=usage or {}
    for key,field in (('gen_ai.usage.input_tokens','prompt_tokens'),('gen_ai.usage.output_tokens','completion_tokens'),('gen_ai.usage.total_tokens','total_tokens')):
        if usage.get(field) is not None:span.set_data(key,usage[field])
    span.set_data('coach.first_token_ms',first_token_ms);span.set_data('coach.frames_sent',frames)
