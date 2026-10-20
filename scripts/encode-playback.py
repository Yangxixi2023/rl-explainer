from pathlib import Path
import json
from PIL import Image

project = Path(__file__).resolve().parent.parent
folder = project / 'output' / 'playback-frames'
frames = json.loads((folder / 'index.json').read_text(encoding='utf-8'))
images = []
for frame in frames:
    image = Image.open(folder / frame['file']).convert('RGB')
    if image.width > 1440:
        image = image.resize((1440, round(image.height * 1440 / image.width)), Image.Resampling.LANCZOS)
    images.append(image)
durations = [max(1, round((b['timestamp'] - a['timestamp']) * 1000)) for a, b in zip(frames, frames[1:])]
durations.append(650)
output = project / 'output' / 'motion-preview.webp'
images[0].save(output, format='WEBP', save_all=True, append_images=images[1:], duration=durations, loop=0, quality=90, method=0)
print(json.dumps({'path': str(output), 'frames': len(images), 'duration_ms': sum(durations), 'bytes': output.stat().st_size}))
