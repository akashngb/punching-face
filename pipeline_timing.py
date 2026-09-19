"""Persist measured pipeline stages independently of the capture/cache hash."""
import json,time
from face_pipeline import atomic

class PipelineTimer:
    def __init__(self,folder):
        self.folder=folder;self.started=time.perf_counter();self.stage_started=self.started;self.stage=None;self.stages=[]
        previous=json.loads((folder/'timing.json').read_text()) if (folder/'timing.json').exists() else {}
        self.requested=previous.get('requestedAt',time.time()) if previous.get('status')=='running' else time.time()
        self.attempts=previous.get('previousAttempts',[])
        self.startup=max(0,time.time()-self.requested)
    def mark(self,stage):
        now=time.perf_counter()
        if self.stage:self.stages.append({'stage':self.stage,'seconds':round(now-self.stage_started,3)})
        self.stage=stage;self.stage_started=now
        return self.save('running')
    def save(self,state):
        result={'status':state,'requestedAt':self.requested,'startupSeconds':round(self.startup,3),'reconstructionSeconds':round(self.startup+time.perf_counter()-self.started,3),'stages':self.stages.copy(),'previousAttempts':self.attempts}
        if state=='running':result.update(activeStage=self.stage,stageStartedAt=time.time()-(time.perf_counter()-self.stage_started))
        atomic(self.folder/'timing.json',result);return result
    def finish(self,state='complete'):
        if self.stage:self.stages.append({'stage':self.stage,'seconds':round(time.perf_counter()-self.stage_started,3)})
        self.stage=None
        return self.save(state)
