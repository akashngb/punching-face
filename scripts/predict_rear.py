"""Generate one cached rear-head appearance prediction for a photo capture.

Uses the user's configured API key through the bundled Image Generation CLI.
The generated image is a modeling prediction, never photographic evidence.
"""

from pathlib import Path
import argparse, json, os, subprocess, sys, hashlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from openai_capture import config

PROMPT = (
    'Use case: identity-preserve\n'
    'Asset type: orthographic rear-view texture reference for a complete 3D head '
    'model.\n'
    'Input images 1, 2 and 3 are actual FRONT, LEFT and RIGHT photographic '
    'references of the SAME consenting adult. Preserve the observed dark wavy '
    'hairstyle, short tapered sides, hair color, hair density, skin tone and '
    'head proportions.\n'
    "Generate ONE plausible rear view of this same person's head, turned exactly "
    '180 degrees away. Infer the unseen occiput, rear hairstyle, natural crown '
    'whorl and tapered nape consistently from these front and side photos. This '
    'is an appearance prediction for unseen surfaces.\n'
    'Composition: strict straight-on back view, zero tilt, orthographic lens, '
    'centered isolated head. Show the full crown, both ears, occiput and bottom '
    'hairline. Crop immediately below the base of the skull; no shoulders, '
    'torso, clothing or long neck. Head occupies about 85 percent of the image '
    'height, with modest equal whitespace.\n'
    'Lighting: diffuse neutral soft illumination suitable for a photographic '
    'texture, no directional cast shadow, no dramatic rim light.\n'
    'Background: pure uniform white. No contact shadow. No checkerboard, '
    'collage, multiple views, text, labels or watermark. No eyes, nose, mouth, '
    'or frontal facial features visible from behind. Keep the hairstyle '
    'recognizable and restrained, do not grow it into a new hairstyle.'
)


def predict(folder):
    dest = folder / 'rear-prediction'
    dest.mkdir(exist_ok=True)
    output = dest / 'rear.png'
    meta = dest / 'prediction.json'
    signature = hashlib.sha256((folder / 'capture.json').read_bytes()).hexdigest()
    if (
        output.exists()
        and meta.exists()
        and json.loads(meta.read_text()).get('captureHash') == signature
    ):
        return json.loads(meta.read_text())
    key, _ = config()
    if not key:
        raise ValueError('OpenAI is required to predict the rear appearance.')
    cli = Path(
        os.environ.get(
            'PUNCHING_FACE_IMAGE_CLI',
            str(Path.home() / '.codex/skills/.system/imagegen/scripts/image_gen.py'),
        )
    )
    if not cli.is_file():
        raise ValueError(
            'Install the Image Generation skill or set PUNCHING_FACE_IMAGE_CLI to its image_gen.py CLI.'
        )
    frames = [
        f
        for f in json.loads((folder / 'capture.json').read_text())['frames']
        if f.get('landmarks')
    ]
    images = []
    for label, angle in [('front', 0), ('left', -45), ('right', 45)]:
        frame = min(frames, key=lambda f: abs(f['yaw'] - angle))
        im = Image.open(folder / 'images' / frame['filename']).convert('RGBA')
        im = im.crop(im.getbbox())
        path = dest / (label + '.png')
        im.save(path)
        images.extend(['--image', str(path)])
    prompt = dest / 'prompt.txt'
    prompt.write_text(PROMPT)
    env = os.environ.copy()
    env['OPENAI_API_KEY'] = key
    command = [
        sys.executable,
        str(cli),
        'edit',
        '--model',
        'gpt-image-2',
        *images,
        '--prompt-file',
        str(prompt),
        '--size',
        '1024x1024',
        '--quality',
        'high',
        '--background',
        'opaque',
        '--out',
        str(output),
    ]
    with (dest / 'generation.log').open('w') as log:
        try:
            result = subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300
            )
        except subprocess.TimeoutExpired:
            raise ValueError(
                'Rear image prediction timed out; captured face photos remain intact.'
            )
    if result.returncode or not output.exists():
        raise ValueError(
            (
                'Rear appearance generation failed. Captured face photos remain '
                'intact; inspect the local generation log.'
            )
        )
    record = {
        'model': 'gpt-image-2',
        'captureHash': signature,
        'source': 'Generated prediction from front and side photographs, not an observed rear view.',
        'framesSent': 3,
        'image': 'rear-prediction/rear.png',
        'prompt': 'rear-prediction/prompt.txt',
    }
    meta.write_text(json.dumps(record, indent=2))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    print(json.dumps(predict(args.folder.resolve()), indent=2))
