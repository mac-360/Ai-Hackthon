"""Video translation for the ASL app.

Reads a video file, detects + classifies the hand in sampled frames, and
decodes the per-frame predictions into text — a held sign becomes one letter,
brief noise between signs is dropped.
"""
import numpy as np
from PIL import Image

try:
    import cv2
    CV2_OK = True
except Exception:
    CV2_OK = False

from hand_detection import HandLocator


def decode_sequence(preds, confs, min_run=3, min_conf=0.35):
    """Collapse a per-frame prediction stream into text.

    A label that persists for >= `min_run` consecutive frames (with mean
    confidence >= `min_conf`) emits one letter; shorter runs are treated as
    noise and dropped. `space` and `del` behave as in the live speller.
    """
    text, i, n = '', 0, len(preds)
    while i < n:
        j = i
        while j < n and preds[j] == preds[i]:
            j += 1
        run_conf = float(np.mean(confs[i:j])) if j > i else 0.0
        lab = preds[i]
        if (j - i) >= min_run and run_conf >= min_conf and lab != 'nothing':
            if lab == 'space':
                text += ' '
            elif lab == 'del':
                text = text[:-1]
            else:
                text += lab
        i = j
    return text


def translate_video(video_path, classify_fn, force_box=False,
                     max_frames=170, progress=None):
    """Translate a fingerspelling video into text.

    `classify_fn(pil_image)` must return a tuple whose first two elements are
    (label, confidence). `progress(fraction)` is an optional callback.
    Returns (translated_text, per_frame_predictions).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return '', []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, total // max_frames) if total else 2
    target = max(1, total // stride) if total else max_frames
    locator = HandLocator(video=True, force_box=force_box)
    preds, confs, idx, done = [], [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            box, status = locator.locate(frame)
            letter, conf = 'nothing', 0.0
            if status != 'none':
                x1, y1, x2, y2 = box
                crop = frame[y1:y2, x1:x2]
                if crop.size:
                    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    out = classify_fn(pil)
                    letter, conf = out[0], out[1]
            preds.append(letter)
            confs.append(conf)
            done += 1
            if progress:
                progress(min(1.0, done / target))
        idx += 1
    cap.release()
    return decode_sequence(preds, confs), preds
