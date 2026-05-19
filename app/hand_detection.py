"""Hand localisation for the ASL app.

Finds the hand region in a frame so only the hand is classified:
  * MediaPipe Hands if explicitly enabled (best);
  * else **background subtraction** once a reference background is captured —
    detects the hand by what *changed*, so it works even when the hand and
    the wall are the same colour, and even when the hand is held still;
  * else OpenCV skin-colour segmentation (with the face removed);
  * else a centred guide box.

`HandLocator.locate()` returns (box, status) where status is:
  'detected' — a hand was actually found
  'none'     — detection ran but found no hand
  'box'      — fixed-box mode (detection bypassed)
"""
import importlib.util
import os

import numpy as np

try:
    import cv2
    CV2_OK = True
except Exception:
    CV2_OK = False

_MP = None
MP_OK = (os.environ.get('ASL_ENABLE_MEDIAPIPE') == '1' and
         importlib.util.find_spec('mediapipe') is not None)


def _load_mediapipe():
    """Import MediaPipe only when explicitly enabled.

    Some hosted Python runtimes segfault while importing MediaPipe/TensorFlow.
    Keeping it lazy makes the app fall back to OpenCV instead of crashing.
    """
    global _MP, MP_OK
    if not MP_OK:
        return None
    if _MP is None:
        try:
            import mediapipe as mp
            _MP = mp
        except Exception:
            MP_OK = False
            return None
    return _MP

_KERNEL = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
           if CV2_OK else None)


def square_box(x1, y1, x2, y2, w, h):
    """Expand a (possibly float) box to a clamped integer square."""
    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = side / 2.0
    return (max(0, int(cx - half)), max(0, int(cy - half)),
            min(w, int(cx + half)), min(h, int(cy + half)))


def skin_mask(bgr, face_cascade=None):
    """Binary mask of skin-coloured pixels (YCrCb); detected faces erased."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    if face_cascade is not None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        try:
            faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(70, 70))
        except Exception:
            faces = []
        for (fx, fy, fw, fh) in faces:
            m = int(0.30 * fw)
            cv2.rectangle(mask, (fx - m, fy - m),
                          (fx + fw + m, fy + fh + m), 0, -1)
    return mask


def largest_box(mask, w, h, lo=0.012, hi=0.85):
    """Clean a mask and return the bounding box of its largest blob, or None
    if no blob falls within [lo, hi] x frame-area."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = float(w * h)
    cnts = [c for c in cnts if lo * area < cv2.contourArea(c) < hi * area]
    if not cnts:
        return None
    x, y, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return (x, y, x + bw, y + bh)


def _gray_blur(bgr):
    return cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (7, 7), 0)


class HandLocator:
    """Unified hand finder. `locate()` -> ((x1,y1,x2,y2), status).

    Call `capture_background(frame)` (with the hand out of view) to switch on
    background-subtraction detection — colour-independent, so it separates the
    hand from a same-coloured wall.
    """

    def __init__(self, video=True, force_box=False):
        self.mp_hands = None
        self.face = None
        self.prev = None
        self.background = None          # reference frame (grayscale, blurred)
        if force_box or not CV2_OK:
            self.method = 'box'
        else:
            mp = _load_mediapipe()
            if mp is not None:
                self.mp_hands = mp.solutions.hands.Hands(
                    static_image_mode=not video, max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
                self.method = 'mediapipe'
                return
            self.face = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.method = 'skin'

    # --- background calibration ---------------------------------------
    def capture_background(self, bgr):
        """Store the current (hand-free) frame as the reference background."""
        self.background = _gray_blur(bgr)

    def clear_background(self):
        self.background = None

    @property
    def has_background(self):
        return self.background is not None

    # --- detection ----------------------------------------------------
    def _smooth(self, box):
        if self.prev is not None:
            a = 0.5
            box = tuple(int(a * n + (1 - a) * p)
                        for n, p in zip(box, self.prev))
        self.prev = tuple(box)
        return tuple(box)

    def _pad(self, box, frac=0.12):
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        return (x1 - frac * bw, y1 - frac * bh,
                x2 + frac * bw, y2 + frac * bh)

    def locate(self, bgr, roi_frac=0.6):
        h, w = bgr.shape[:2]
        raw = None

        if self.mp_hands is not None:
            res = self.mp_hands.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
                xs = [p.x * w for p in lm]
                ys = [p.y * h for p in lm]
                bw, bh = max(xs) - min(xs), max(ys) - min(ys)
                raw = (min(xs) - 0.40 * bw, min(ys) - 0.40 * bh,
                       max(xs) + 0.40 * bw, max(ys) + 0.40 * bh)

        elif self.background is not None:
            # background subtraction — what CHANGED vs the empty scene.
            diff = cv2.absdiff(_gray_blur(bgr), self.background)
            _, dmask = cv2.threshold(diff, 32, 255, cv2.THRESH_BINARY)
            # refine with skin colour: keep only the skin-coloured change
            # (removes shadows, sleeves and lighting wobble) — and the
            # unchanged wall is excluded because diff is zero there.
            combined = cv2.bitwise_and(dmask, skin_mask(bgr, self.face))
            box = largest_box(combined, w, h, lo=0.010, hi=0.92)
            if box is None:                       # fall back to motion alone
                box = largest_box(dmask, w, h, lo=0.010, hi=0.92)
            if box is not None:
                raw = self._pad(box)

        elif self.face is not None:
            box = largest_box(skin_mask(bgr, self.face), w, h,
                              lo=0.012, hi=0.60)
            if box is not None:
                raw = self._pad(box)

        if raw is not None:
            return self._smooth(square_box(*raw, w, h)), 'detected'

        # nothing found — return a centred guide box
        self.prev = None
        side = int(min(h, w) * roi_frac)
        cx, cy = w // 2, h // 2
        box = (cx - side // 2, cy - side // 2,
               cx + side // 2, cy + side // 2)
        return box, ('box' if self.method == 'box' else 'none')
