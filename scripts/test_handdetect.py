"""Verifies hand_detection.py — skin detection, background subtraction, and
the fixed-box path — on real images. Run from the repo root:
    python scripts/test_handdetect.py"""
import os, sys, random
import numpy as np
from PIL import Image
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'app'))   # hand_detection.py lives in app/

from hand_detection import HandLocator, square_box, skin_mask, largest_box, \
    MP_OK, CV2_OK

print(f'CV2_OK={CV2_OK}  MP_OK={MP_OK}')
ok = True


def bgr_of(path):
    return cv2.cvtColor(np.array(Image.open(path).convert('RGB')),
                        cv2.COLOR_RGB2BGR)


# square_box sanity
b = square_box(10, 30, 90, 70, 200, 200)
assert len(b) == 4 and b[2] > b[0] and b[3] > b[1], b
print('square_box: OK', b)

base = next((c for c in [
    os.path.join(ROOT, 'data', 'competition_data', 'competition_data'),
    os.path.join(ROOT, 'data', 'competition_data'),
    os.path.join(ROOT, 'data')]
    if os.path.isdir(os.path.join(c, 'train'))), None)
if base is None:
    print('no local data — skipping image tests')
    raise SystemExit(0)
train = os.path.join(base, 'train')
random.seed(0)
classes = sorted(os.listdir(train))
paths = [os.path.join(train, c, random.choice(os.listdir(os.path.join(train, c))))
         for c in random.sample(classes, 14)]

# --- 1. skin-segmentation path ---
loc = HandLocator(video=False)
print(f'\\n[1] skin path (method={loc.method})')
n_valid = 0
for p in paths:
    bgr = bgr_of(p)
    (x1, y1, x2, y2), status = loc.locate(bgr)
    H, W = bgr.shape[:2]
    valid = 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
    n_valid += valid
    if not valid:
        ok = False
print(f'    valid boxes: {n_valid}/{len(paths)}')

# --- 2. background-subtraction path ---
print('\\n[2] background-subtraction path')
loc2 = HandLocator(video=False)
bg = bgr_of(paths[0])
loc2.capture_background(bg)
assert loc2.has_background, 'background not stored'
# same frame as background -> nothing changed -> should be "none"
_, s_same = loc2.locate(bg)
print(f'    same-as-background frame -> status={s_same}  '
      f'(expect "none")')
if s_same != 'none':
    ok = False
# a different frame -> something changed -> a box, valid
(x1, y1, x2, y2), s_diff = loc2.locate(bgr_of(paths[1]))
H, W = bgr_of(paths[1]).shape[:2]
valid = 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
print(f'    different frame -> status={s_diff}  box=({x1},{y1},{x2},{y2})  '
      f'valid={valid}')
if not valid:
    ok = False
loc2.clear_background()
assert not loc2.has_background, 'clear_background failed'
print('    capture / clear background: OK')

# --- 3. fixed-box path ---
(fx1, fy1, fx2, fy2), s_box = HandLocator(video=False, force_box=True).locate(bg)
print(f'\\n[3] force-box path -> status={s_box}  box=({fx1},{fy1},{fx2},{fy2})')
if s_box != 'box':
    ok = False

print('\\nRESULT:', 'PASS' if ok else 'FAIL')
