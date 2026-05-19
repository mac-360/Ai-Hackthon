"""End-to-end test of video translation: unit-checks the decoder, then builds
a synthetic clip from dataset images and checks translate_video reads it and
decodes the letters. Run from the repo root:  python scripts/test_video.py"""
import os, sys
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import timm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'app'))
from video_translate import translate_video, decode_sequence

ok = True

# --- 1. decoder unit checks ------------------------------------------------
assert decode_sequence(['nothing']*2 + ['A']*6 + ['nothing']*3 + ['B']*6,
                       [0.9]*17) == 'AB'
assert decode_sequence(['A']*6 + ['space']*4 + ['B']*6, [0.9]*16) == 'A B'
assert decode_sequence(['A']*2, [0.9]*2) == ''           # run too short -> dropped
assert decode_sequence(['A']*6, [0.1]*6) == ''           # low confidence -> dropped
print('[1] decode_sequence unit checks: OK')

# --- 2. end-to-end on a synthetic clip ------------------------------------
base = next((c for c in [
    os.path.join(ROOT, 'data', 'competition_data', 'competition_data'),
    os.path.join(ROOT, 'data', 'competition_data'),
    os.path.join(ROOT, 'data')]
    if os.path.isdir(os.path.join(c, 'train'))), None)
if base is None:
    print('no local data — skipping end-to-end video test')
    raise SystemExit(0)
train = os.path.join(base, 'train')

letters = ['A', 'B', 'C']
clip = os.path.join(ROOT, '_test_clip.avi')
vw = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*'MJPG'), 10, (200, 200))
for L in letters:
    for f in sorted(os.listdir(os.path.join(train, L)))[:18]:
        img = cv2.imread(os.path.join(train, L, f))
        if img is not None:
            vw.write(cv2.resize(img, (200, 200)))
vw.release()

# load the model + a classify function
ckpt = torch.load(os.path.join(ROOT, 'app', 'best_model.pt'),
                  map_location='cpu', weights_only=False)
CLASSES = ckpt['classes']
model = timm.create_model('convnext_tiny', pretrained=False,
                          num_classes=len(CLASSES))
model.load_state_dict(ckpt['model'])
model.eval()
tfm = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                 T.ToTensor(),
                 T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

@torch.no_grad()
def classify_fn(pil):
    p = F.softmax(model(tfm(pil.convert('RGB')).unsqueeze(0)), 1)[0].numpy()
    i = int(p.argmax())
    return CLASSES[i], float(p[i])

text, preds = translate_video(clip, classify_fn, force_box=False)
os.remove(clip)

print(f'[2] synthetic clip ({len(preds)} frames) decoded -> {text!r}')
found = sum(L in text for L in letters)
print(f'    letters recovered: {found}/3')
if found < 2:
    ok = False

print('\nRESULT:', 'PASS' if ok else 'FAIL')
