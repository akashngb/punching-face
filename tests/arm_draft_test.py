"""Exercise capture persistence without training or claiming a personal scan."""

import base64, io, json, urllib.request, urllib.error
from pathlib import Path
from PIL import Image

root = Path(__file__).resolve().parents[1]
image = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
buffer = io.BytesIO()
image.save(buffer, format='PNG')
raw = buffer.getvalue()
encoded = 'data:image/png;base64,' + base64.b64encode(raw).decode()
frame = {
    'image': encoded,
    'mask': encoded,
    'original': encoded,
    'pose': [{'x': 0.5, 'y': 0.5, 'z': 0, 'visibility': 1}] * 33,
    'hand': [{'x': 0.5, 'y': 0.5, 'z': 0}] * 21,
    'orientationBin': 0,
}


def post(route, data):
    request = urllib.request.Request(
        'http://127.0.0.1:5174' + route,
        data=json.dumps(data).encode(),
        headers={'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


code, result = post(
    '/api/arm-draft',
    {
        'side': 'left',
        'frames': [frame],
        'testFixture': 'single blank frame persistence test; not a personal capture',
    },
)
assert code == 201, (code, result)
folder = root / '.local/arm-captures' / result['id']
manifest = json.loads((folder / 'capture.json').read_text())
state = json.loads((folder / 'status.json').read_text())
assert state['status'] == 'captured'
assert len(manifest['frames']) == 1
assert (folder / 'originals/frame_0000.png').read_bytes() == raw
assert not (folder / 'pipeline.log').exists()
code, error = post('/api/arm-train', {'id': result['id']})
assert code == 422 and '12' in error['error']
with urllib.request.urlopen('http://127.0.0.1:5174/api/arm-drafts') as response:
    drafts = json.load(response)['drafts']
assert all(draft['id'] != result['id'] for draft in drafts)
print(
    json.dumps(
        {
            'draftId': result['id'],
            'savedWithoutTraining': True,
            'originalBytesPreserved': True,
            'incompleteDraftTrainingRejected': True,
            'testFixtureExcludedFromUserList': True,
        }
    )
)
