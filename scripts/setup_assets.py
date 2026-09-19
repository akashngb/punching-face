"""Fetch the public assets used by this local prototype. Never uploads captures."""
from pathlib import Path
from urllib.request import urlopen
import shutil,zipfile,tarfile,hashlib,platform
ROOT=Path(__file__).resolve().parents[1]
def fetch(url,path):
    path=ROOT/path;path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        temp=path.with_suffix(path.suffix+'.download')
        with urlopen(url,timeout=60) as response,temp.open('wb') as output:shutil.copyfileobj(response,output)
        temp.replace(path)
    return path

base='https://storage.googleapis.com/mediapipe-models'
for name,url in {
    'selfie_multiclass_256x256.tflite':f'{base}/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite',
    'hand_landmarker.task':f'{base}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    'face_landmarker.task':f'{base}/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
    'pose_landmarker.task':f'{base}/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
    'canonical_face_model.obj':'https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj',
}.items():fetch(url,Path('public/models')/name)
(ROOT/'public/vendor').mkdir(parents=True,exist_ok=True)
shutil.copyfile(ROOT/'node_modules/@mediapipe/tasks-vision/vision_bundle.cjs',ROOT/'public/vendor/vision_bundle.cjs')
shutil.copytree(ROOT/'node_modules/@mediapipe/tasks-vision/wasm',ROOT/'public/wasm',dirs_exist_ok=True)
for name in ('LeePerrySmith.glb','Map-COL.jpg'):
    fetch('https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/gltf/LeePerrySmith/'+name,Path('public/reference')/name)
archive=fetch('https://raw.githubusercontent.com/aigc3d/LAM_WebRender/main/asset/arkit/p2-1.zip',Path('.local/sources/lam/p2-1.zip'))
with zipfile.ZipFile(archive) as z:
    for name in ('skin.glb','offset.ply','animation.glb','vertex_order.json'):
        path=ROOT/'.local/sources/lam/p2-1'/name;path.parent.mkdir(exist_ok=True);path.write_bytes(z.read('p2-1/'+name))
fetch('https://raw.githubusercontent.com/aigc3d/LAM_WebRender/main/LICENSE',Path('public/online-face/LICENSE.txt'))
if platform.system()=='Darwin' and platform.machine()=='arm64':
    name='brush-app-aarch64-apple-darwin.tar.xz'
    archive=fetch('https://github.com/ArthurBrussee/brush/releases/download/v0.3.0/'+name,Path('.local/tools/brush')/name)
    expected='65b2631398c839be3c1d4d7160fe2326389dec87830aac0710985e6690a1048c'
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected:raise ValueError('Brush archive checksum mismatch.')
    destination=ROOT/'.local/tools/brush'
    with tarfile.open(archive) as t:
        members=t.getmembers()
        for m in members:
            if not (destination/m.name).resolve().is_relative_to(destination.resolve()) or m.issym() or m.islnk():raise ValueError('Unexpected archive entry.')
        t.extractall(destination,members=members)
print('Public assets ready. Run make_fixture.py, prepare_lam.py, and bake_texture.py next.')
