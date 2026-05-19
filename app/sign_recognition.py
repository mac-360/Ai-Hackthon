"""Dynamic ASL sign recognition.

Recognises moving, word-level ASL signs (HELLO, THANK YOU, YES…) from a short
window of video — MediaPipe Holistic landmarks fed to an LSTM when MediaPipe
is explicitly enabled.

Shared by the data collector (scripts/collect_signs.py), the training notebook
(notebooks/04_dynamic_signs.ipynb) and the app (app/app.py), so the feature
contract and model definition never drift.
"""
import importlib.util
import os

import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    Dynamic sign recognition is optional in the Streamlit app, and native
    MediaPipe imports can segfault in some hosted notebook runtimes.
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

# --- feature contract: every component agrees on these ---
SEQ_LEN = 30                                    # frames per sign window
POSE_N, HAND_N = 33, 21
FEATURE_DIM = (POSE_N + HAND_N + HAND_N) * 3    # pose + 2 hands, x/y/z = 225


# ======================================================================
# Landmark extraction (MediaPipe Holistic)
# ======================================================================
def make_holistic():
    """A MediaPipe Holistic tracker, or None if MediaPipe is unavailable."""
    mp = _load_mediapipe()
    if mp is None:
        return None
    return mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)


def _block(landmarks, n):
    """Flatten one landmark group to n*3 floats; zeros if not detected."""
    if landmarks is None:
        return np.zeros(n * 3, dtype=np.float32)
    return np.array([[p.x, p.y, p.z] for p in landmarks.landmark],
                    dtype=np.float32).flatten()


def extract_landmarks(frame_bgr, holistic):
    """A BGR frame -> a FEATURE_DIM vector: pose (33) + left & right hand
    (21 each), x/y/z. Undetected parts are zero-filled."""
    res = holistic.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return np.concatenate([
        _block(res.pose_landmarks, POSE_N),
        _block(res.left_hand_landmarks, HAND_N),
        _block(res.right_hand_landmarks, HAND_N),
    ]).astype(np.float32)


# ======================================================================
# Sequence augmentation (used by the training notebook)
# ======================================================================
def augment_sequence(seq, rng):
    """Augmented copy of a (SEQ_LEN, FEATURE_DIM) sequence — coordinate
    noise, a small temporal shift, and a 50% horizontal mirror."""
    seq = seq.astype(np.float32).copy()
    seq += rng.normal(0, 0.012, seq.shape).astype(np.float32)      # jitter
    seq = np.roll(seq, int(rng.integers(-3, 4)), axis=0)           # time shift
    if rng.random() < 0.5:                                         # mirror
        s = seq.reshape(seq.shape[0], -1, 3)
        s[..., 0] = 1.0 - s[..., 0]                                # flip x
        pose = s[:, :POSE_N]
        lh = s[:, POSE_N:POSE_N + HAND_N]
        rh = s[:, POSE_N + HAND_N:]
        s = np.concatenate([pose, rh, lh], axis=1)                 # swap hands
        seq = s.reshape(seq.shape[0], -1)
    return seq.astype(np.float32)


# ======================================================================
# Model
# ======================================================================
class SignLSTM(nn.Module):
    """Stacked-LSTM classifier over a window of landmark frames."""

    def __init__(self, n_classes, feature_dim=FEATURE_DIM, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden, num_layers=2,
                            batch_first=True, dropout=0.3)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes))

    def forward(self, x):                       # x: (B, SEQ_LEN, FEATURE_DIM)
        out, _ = self.lstm(x)
        return self.head(out[:, -1])            # classify the final timestep


def load_sign_model(path):
    """Load a trained dynamic-sign checkpoint -> (model, labels)."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    labels = list(ckpt['labels'])
    model = SignLSTM(len(labels), ckpt.get('feature_dim', FEATURE_DIM),
                     ckpt.get('hidden', 128))
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, labels


# ======================================================================
# Live recogniser
# ======================================================================
class SignRecognizer:
    """Feed frames in; once a full SEQ_LEN window is buffered, `predict()`
    returns the current sign and a confidence."""

    def __init__(self, model_path, min_conf=0.6):
        self.model, self.labels = load_sign_model(model_path)
        self.holistic = make_holistic()
        self.window = deque(maxlen=SEQ_LEN)
        self.min_conf = min_conf

    def reset(self):
        self.window.clear()

    def observe(self, frame_bgr):
        """Extract + buffer one frame's landmarks. True when the window fills."""
        if self.holistic is None:
            return False
        self.window.append(extract_landmarks(frame_bgr, self.holistic))
        return len(self.window) == SEQ_LEN

    @torch.no_grad()
    def predict(self):
        """(label, confidence) over the current window, or (None, 0.0)."""
        if len(self.window) < SEQ_LEN:
            return None, 0.0
        x = torch.tensor(np.stack(self.window),
                         dtype=torch.float32).unsqueeze(0)
        probs = F.softmax(self.model(x), dim=1)[0].numpy()
        i = int(probs.argmax())
        return self.labels[i], float(probs[i])
