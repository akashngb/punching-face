"""Integration check: unusable capture fails honestly, without a fake arm."""
import base64,io,json,time
from urllib.request import Request,urlopen
from pathlib import Path
from PIL import Image

buffer=io.BytesIO();Image.new('RGBA',(64,64),(0,0,0,0)).save(buffer,format='PNG')
image='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()
joint={'x':.5,'y':.5,'z':0,'visibility':1}
frames=[{'image':image,'original':image,'mask':image,'pose':[joint]*33,'hand':[joint]*21,'timestamp':i*1000} for i in range(12)]
request=Request('http://127.0.0.1:5174/api/arm-capture',data=json.dumps({'side':'left','frames':frames,'testFixture':'blank images: MUST fail, not a personal scan'}).encode(),headers={'Content-Type':'application/json','Origin':'http://127.0.0.1:5173'})
result=json.load(urlopen(request,timeout=10));deadline=time.monotonic()+30
while time.monotonic()<deadline:
    state=json.load(urlopen('http://127.0.0.1:5174/api/arm-status?id='+result['id'],timeout=10))
    if state['status'] in ('failed','complete'):break
    time.sleep(.5)
assert state['status']=='failed',state
assert 'bundle' not in state,state
assert not (Path(result['path'])/'arm-bundle.json').exists()
print(json.dumps({'test':'blank capture is rejected without a substitute mesh','status':state},indent=2))
