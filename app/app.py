"""The Silent Gap — ASL Sign Detector, Speller & Tutor (Streamlit app).

Recognises American Sign Language fingerspelling from a webcam, a snapshot,
or an uploaded image. It detects the hand, classifies only that region,
spells words (with predictive autocomplete), speaks them aloud, and includes
a Practice mode for learning the ASL alphabet.

Run:  streamlit run app.py
"""
import os
import sys
import io
import glob
import random
import hashlib
import tempfile
import threading
from collections import deque, Counter
from html import escape

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import timm
import streamlit as st
import streamlit.components.v1 as components

# make sibling modules importable however the app is launched
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hand_detection import HandLocator, MP_OK   # hand detection logic
from video_translate import translate_video     # video -> text translation
from sign_recognition import SignRecognizer, SEQ_LEN  # dynamic-sign recognition

# --- optional: real-time video --------------------------------------------
try:
    import av
    import cv2
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    WEBRTC_OK = True
except Exception:
    WEBRTC_OK = False
if os.environ.get('ASL_DISABLE_WEBRTC') == '1':
    WEBRTC_OK = False
if 'streamlit.testing' in sys.modules or 'streamlit.testing.v1' in sys.modules:
    WEBRTC_OK = False

# --- optional: speech-to-text via the browser microphone ------------------
try:
    from streamlit_mic_recorder import speech_to_text
    STT_OK = True
except Exception:
    STT_OK = False

# --- optional: English <-> Urdu translation -------------------------------
try:
    from deep_translator import GoogleTranslator
    TRANSLATE_OK = True
except Exception:
    TRANSLATE_OK = False

st.set_page_config(page_title='ASL Sign Detector', page_icon='🤟', layout='wide')

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
DEFAULT_CLASSES = sorted(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') +
                         ['del', 'nothing', 'space'])
ARCH_CANDIDATES = ['convnext_tiny', 'convnext_base', 'convnextv2_base',
                   'tf_efficientnetv2_s', 'tf_efficientnetv2_m',
                   'swin_base_patch4_window7_224', 'efficientnet_b0']

# All app assets (model, wordlist, reference images) live next to this file,
# so the app works no matter which directory it is launched from.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.join(APP_DIR, 'reference_signs')


def pretty(label):
    return {'del': 'DEL', 'space': 'SPACE', 'nothing': '—'}.get(label, label)


# ==========================================================================
# Model
# ==========================================================================
@st.cache_resource(show_spinner='Loading model...')
def load_model(path):
    """Load a checkpoint; auto-detect architecture, classes and input size."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    classes, arch, img_size, meta = None, None, 224, {}
    if isinstance(ckpt, dict) and isinstance(ckpt.get('model'), dict):
        state = ckpt['model']
        classes = ckpt.get('classes')
        cfg = ckpt.get('cfg') or {}
        arch = cfg.get('model_name')
        img_size = int(cfg.get('img_size', 224) or 224)
        meta = {'epoch': ckpt.get('epoch'), 'val_acc': ckpt.get('val_acc'),
                'metrics': ckpt.get('metrics') or {}, 'cfg': cfg}
    elif isinstance(ckpt, dict) and isinstance(ckpt.get('state_dict'), dict):
        state = ckpt['state_dict']
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError('Unrecognised checkpoint format.')

    n_classes = len(classes) if classes else int(list(state.values())[-1].shape[0])
    model = used = None
    for name in ([arch.split('.')[0]] if arch else []) + ARCH_CANDIDATES:
        if not name:
            continue
        try:
            cand = timm.create_model(name, pretrained=False, num_classes=n_classes)
            cand.load_state_dict(state, strict=True)
            model, used = cand, name
            break
        except Exception:
            continue
    if model is None:
        raise RuntimeError('Could not match the checkpoint to a known architecture.')
    model.eval()
    if not classes:
        classes = DEFAULT_CLASSES if n_classes == 29 else \
            [str(i) for i in range(n_classes)]
    return model, list(classes), used, img_size, meta


def build_transform(img_size):
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@torch.no_grad()
def classify(pil_img):
    """Return (label, confidence, [(label, prob) x5]) using the global model."""
    x = TFM(pil_img.convert('RGB')).unsqueeze(0)
    probs = F.softmax(MODEL(x), dim=1)[0].cpu().numpy()
    order = probs.argsort()[::-1]
    return CLASSES[order[0]], float(probs[order[0]]), \
        [(CLASSES[i], float(probs[i])) for i in order[:5]]


# ==========================================================================
# Predictive autocomplete
# ==========================================================================
@st.cache_data(show_spinner=False)
def load_wordlist(path='wordlist.txt'):
    words, seen = [], set()
    if os.path.exists(path):
        for line in open(path, encoding='utf-8'):
            w = line.strip().upper()
            if w and not w.startswith('#') and w.isalpha() and w not in seen:
                seen.add(w)
                words.append(w)
    return words or ['HELLO', 'HELP', 'YES', 'NO', 'THANKS', 'PLEASE', 'NAME']


def predict_words(text, wordlist, k=3):
    """Suggest completions for the word currently being spelled."""
    prefix = (text or '').split(' ')[-1].strip().upper()
    if not prefix:
        return []
    out = []
    for w in wordlist:
        if w.startswith(prefix) and w != prefix:
            out.append(w)
            if len(out) >= k:
                break
    return out


# ==========================================================================
# Detect-then-classify (shared by Snapshot, Upload and Practice)
# ==========================================================================
def detect_and_classify(pil_img, force_box=False):
    """Locate the hand, crop to it, classify. Returns
    (letter, conf, top5, crop_image, note)."""
    crop_img, note = pil_img, 'whole image'
    if WEBRTC_OK:                       # cv2 available -> run hand detection
        bgr = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
        box, status = HandLocator(video=False, force_box=force_box).locate(bgr)
        if status in ('detected', 'box'):
            x1, y1, x2, y2 = box
            sub = bgr[y1:y2, x1:x2]
            if sub.size:
                crop_img = Image.fromarray(cv2.cvtColor(sub, cv2.COLOR_BGR2RGB))
                note = ('hand detected and cropped' if status == 'detected'
                        else 'fixed centre crop')
        else:
            note = 'no hand detected — classified the whole image'
    letter, conf, top5 = classify(crop_img)
    return letter, conf, top5, crop_img, note


# ==========================================================================
# Real-time video processor — detect hand, classify, spell, autocomplete
# ==========================================================================
class ASLProcessor:
    def __init__(self, force_box=False, wordlist=None):
        self.locator = HandLocator(video=True, force_box=force_box)
        self.wordlist = wordlist or []
        self.recent = deque(maxlen=10)
        self.word = ''
        self.letter = ''
        self.conf = 0.0
        self.suggestions = []
        self.cooldown = 0
        self.commit_votes = 7        # frames of agreement needed to commit
        self.min_conf = 0.40
        self.roi_frac = 0.60
        self.capture_request = False
        self.clear_request = False
        self.lock = threading.Lock()

    def _commit(self, letter):
        if letter == 'space':
            self.word += ' '
        elif letter == 'del':
            self.word = self.word[:-1]
        elif letter != 'nothing':
            self.word += letter

    def recv(self, frame):
        img = frame.to_ndarray(format='bgr24')
        h, w = img.shape[:2]

        with self.lock:
            cap, clr = self.capture_request, self.clear_request
            self.capture_request = self.clear_request = False
        if clr:
            self.locator.clear_background()
        if cap:
            self.locator.capture_background(img)

        (x1, y1, x2, y2), status = self.locator.locate(img, roi_frac=self.roi_frac)

        if status == 'none':
            with self.lock:
                self.letter, self.conf = '', 0.0
                self.recent.clear()
            cv2.rectangle(img, (x1, y1), (x2, y2), (60, 60, 230), 2)
            cv2.putText(img, 'no hand detected', (x1 + 8, max(22, y1 - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 230), 2,
                        cv2.LINE_AA)
        else:
            try:
                crop = img[y1:y2, x1:x2]
                if crop.size:
                    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    letter, conf, _ = classify(pil)
                else:
                    letter, conf = '', 0.0
            except Exception:
                letter, conf = '', 0.0
            with self.lock:
                self.letter, self.conf = letter, conf
                if letter:
                    self.recent.append(letter)
                if self.cooldown > 0:
                    self.cooldown -= 1
                elif len(self.recent) == self.recent.maxlen:
                    top, votes = Counter(self.recent).most_common(1)[0]
                    if votes >= self.commit_votes and conf >= self.min_conf \
                            and top != 'nothing':
                        self._commit(top)
                        self.cooldown = 16
            colour = (0, 220, 0) if status == 'detected' else (0, 180, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
            tag = (f'{pretty(self.letter)}  {self.conf*100:.0f}%'
                   if self.letter else '...')
            cv2.rectangle(img, (x1, max(0, y1 - 42)), (x1 + 330, y1),
                          (0, 0, 0), -1)
            cv2.putText(img, tag, (x1 + 8, max(22, y1 - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, colour, 2, cv2.LINE_AA)

        with self.lock:
            word = self.word
            self.suggestions = predict_words(word, self.wordlist)
            sugg = list(self.suggestions)
        bar = 'WORD: ' + (word or '_')
        if sugg:
            bar += '   > ' + sugg[0]
        cv2.rectangle(img, (0, h - 54), (w, h), (0, 0, 0), -1)
        cv2.putText(img, bar, (14, h - 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.95, (255, 255, 255), 2, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(img, format='bgr24')


# ==========================================================================
# Dynamic-sign video processor — recognise moving signs, build a sentence
# ==========================================================================
class SignWordProcessor:
    """webrtc processor: buffers landmark frames, recognises a dynamic sign
    via the LSTM, and appends confident signs to a sentence."""

    def __init__(self, model_path):
        self.recognizer = SignRecognizer(model_path)
        self.sentence = []
        self.current = ''
        self.conf = 0.0
        self.cooldown = 0
        self.lock = threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format='bgr24')
        h, w = img.shape[:2]

        ready = self.recognizer.observe(img)
        word, conf = self.recognizer.predict() if ready else (None, 0.0)
        with self.lock:
            if word:
                self.current, self.conf = word, conf
            if self.cooldown > 0:
                self.cooldown -= 1
            elif word and conf >= 0.85:          # commit a confident sign
                if not self.sentence or self.sentence[-1] != word:
                    self.sentence.append(word)
                    self.cooldown = 20
            cur, cf, sent = self.current, self.conf, list(self.sentence)

        buffering = len(self.recognizer.window) < SEQ_LEN
        tag = ('buffering...' if buffering else
               (f'{cur.replace("_", " ")}  {cf*100:.0f}%' if cur else '—'))
        cv2.rectangle(img, (0, 0), (w, 46), (0, 0, 0), -1)
        cv2.putText(img, 'SIGN: ' + tag, (14, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 230, 0), 2, cv2.LINE_AA)
        line = ' '.join(s.replace('_', ' ') for s in sent) or '_'
        cv2.rectangle(img, (0, h - 46), (w, h), (0, 0, 0), -1)
        cv2.putText(img, 'SENTENCE: ' + line, (14, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(img, format='bgr24')


# ==========================================================================
# Text-to-speech (browser Web Speech API)
# ==========================================================================
def speak(text, lang='en-US'):
    text = (text or '').replace('\\', ' ').replace('"', ' ').replace("'", ' ')
    if not text.strip():
        return
    components.html(
        f"""<script>
        const u = new SpeechSynthesisUtterance("{text}");
        u.lang = "{lang}";
        u.rate = 0.9;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(u);
        </script>""", height=0)


@st.cache_data(show_spinner=False)
def translate_text(text, src, tgt):
    """Translate `text` between languages (e.g. 'en' <-> 'ur')."""
    return GoogleTranslator(source=src, target=tgt).translate(text) or ''


def pct(value):
    if value is None:
        return '—'
    try:
        return f'{float(value) * 100:.2f}%'
    except Exception:
        return '—'


def display_model_name(arch):
    if arch == 'convnext_tiny':
        return 'Shift-robust ConvNeXt-Tiny'
    return (arch or 'ASL classifier').replace('_', ' ').title()


def inject_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');
    :root {
        --ink: #172026;
        --muted: #64727d;
        --line: #dce4ea;
        --paper: #ffffff;
        --wash: #f6f8fb;
        --blue: #1769aa;
        --teal: #0f8b8d;
        --green: #238636;
        --amber: #b7791f;
        --red: #b42318;
    }
    html, body, [class*="css"] {
        font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .block-container {
        max-width: 1240px;
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
    }
    [data-testid="stSidebar"] {
        background: #f5f8fb;
        border-right: 1px solid var(--line);
    }
    .hero {
        background: linear-gradient(135deg, #102338 0%, #0d4f5f 52%, #166a6f 100%);
        color: white;
        border-radius: 8px;
        padding: 1.35rem 1.45rem;
        margin-bottom: 1rem;
        box-shadow: 0 18px 44px rgba(16, 35, 56, .18);
    }
    .hero h1 {
        margin: 0 0 .25rem 0;
        font-size: clamp(1.8rem, 4vw, 3rem);
        line-height: 1.02;
        letter-spacing: 0;
    }
    .hero p {
        margin: 0;
        max-width: 820px;
        color: rgba(255, 255, 255, .86);
        font-size: 1.02rem;
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: .45rem;
        margin-top: .9rem;
    }
    .badge {
        border: 1px solid rgba(255, 255, 255, .25);
        background: rgba(255, 255, 255, .12);
        color: white;
        border-radius: 999px;
        padding: .34rem .65rem;
        font-size: .82rem;
        font-weight: 650;
    }
    .section-title {
        margin: .2rem 0 .8rem 0;
    }
    .section-title .kicker {
        color: var(--teal);
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin-bottom: .12rem;
    }
    .section-title h2 {
        margin: 0;
        color: var(--ink);
        font-size: 1.45rem;
        letter-spacing: 0;
    }
    .section-title p {
        margin: .35rem 0 0 0;
        color: var(--muted);
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: .65rem;
        margin: .75rem 0 1rem;
    }
    .status-card, .result-card, .empty-card {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: .9rem;
        box-shadow: 0 10px 28px rgba(23, 32, 38, .06);
    }
    .status-card .label {
        color: var(--muted);
        font-size: .78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .06em;
    }
    .status-card .value {
        color: var(--ink);
        font-size: 1.18rem;
        font-weight: 800;
        margin-top: .15rem;
        overflow-wrap: anywhere;
    }
    .status-card .caption {
        color: var(--muted);
        font-size: .82rem;
        margin-top: .18rem;
    }
    .pill-row {
        display: flex;
        gap: .42rem;
        flex-wrap: wrap;
        margin: .35rem 0 .75rem;
    }
    .pill {
        border: 1px solid var(--line);
        background: #f9fbfd;
        color: #31424f;
        border-radius: 999px;
        padding: .3rem .58rem;
        font-size: .8rem;
        font-weight: 650;
    }
    .pill.good { border-color: #b7e2c0; color: var(--green); background: #f1fbf3; }
    .pill.warn { border-color: #efd08c; color: var(--amber); background: #fff9e8; }
    .prediction {
        text-align: center;
        padding: .75rem 0 .55rem;
    }
    .prediction .label {
        color: var(--muted);
        font-size: .78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .08em;
    }
    .prediction .sign {
        color: var(--ink);
        font-size: clamp(4rem, 11vw, 6.2rem);
        line-height: .95;
        font-weight: 850;
        margin-top: .12rem;
    }
    .confidence-track {
        height: .62rem;
        background: #e8edf2;
        border-radius: 999px;
        overflow: hidden;
        margin: .65rem 0 .25rem;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--teal), var(--blue));
        border-radius: 999px;
    }
    .prob-row {
        display: grid;
        grid-template-columns: 3.3rem 1fr 3.8rem;
        gap: .55rem;
        align-items: center;
        margin: .42rem 0;
        color: #2d3b45;
        font-size: .9rem;
    }
    .prob-track {
        height: .48rem;
        background: #e9eef3;
        border-radius: 999px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        background: #1769aa;
        border-radius: 999px;
    }
    .empty-card {
        border-style: dashed;
        background: #fbfcfe;
        color: var(--muted);
    }
    .large-output {
        background: #f5f9fb;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem;
        color: var(--ink);
        font-size: 1.45rem;
        font-weight: 700;
        overflow-wrap: anywhere;
    }
    .urdu {
        font-family: 'Noto Nastaliq Urdu', serif;
        line-height: 2.6;
    }
    div[data-testid="stTabs"] button {
        font-weight: 650;
    }
    @media (max-width: 820px) {
        .metric-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    @media (max-width: 540px) {
        .metric-grid {
            grid-template-columns: 1fr;
        }
        .hero {
            padding: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def section_header(kicker, title, body=None):
    body_html = f'<p>{escape(body)}</p>' if body else ''
    st.markdown(
        f"""<div class="section-title">
        <div class="kicker">{escape(kicker)}</div>
        <h2>{escape(title)}</h2>
        {body_html}
        </div>""",
        unsafe_allow_html=True,
    )


def metric_cards(items):
    cards = []
    for label, value, caption in items:
        cards.append(
            f"""<div class="status-card">
            <div class="label">{escape(label)}</div>
            <div class="value">{escape(str(value))}</div>
            <div class="caption">{escape(caption)}</div>
            </div>"""
        )
    st.markdown('<div class="metric-grid">' + ''.join(cards) + '</div>',
                unsafe_allow_html=True)


def dependency_pills():
    pills = [
        ('Webcam ready' if WEBRTC_OK else 'Webcam fallback', WEBRTC_OK),
        ('MediaPipe hand tracking' if MP_OK else 'OpenCV hand detection', MP_OK),
        ('Microphone STT' if STT_OK else 'Typed input only', STT_OK),
        ('Translation ready' if TRANSLATE_OK else 'Translation offline', TRANSLATE_OK),
    ]
    html = ''.join(
        f'<span class="pill {"good" if ok else "warn"}">{escape(label)}</span>'
        for label, ok in pills
    )
    st.markdown(f'<div class="pill-row">{html}</div>', unsafe_allow_html=True)


def empty_state(title, body):
    st.markdown(
        f"""<div class="empty-card">
        <strong>{escape(title)}</strong><br>
        {escape(body)}
        </div>""",
        unsafe_allow_html=True,
    )


def probability_bars(top5):
    rows = []
    for sign, prob in top5:
        width = max(0.0, min(100.0, float(prob) * 100.0))
        rows.append(
            f"""<div class="prob-row">
            <strong>{escape(pretty(sign))}</strong>
            <div class="prob-track"><div class="prob-fill" style="width:{width:.1f}%"></div></div>
            <span>{width:.1f}%</span>
            </div>"""
        )
    st.markdown(''.join(rows), unsafe_allow_html=True)


def show_result(pil_img, caption, force_box=False):
    """Classify a single image and render the result."""
    letter, conf, top5, crop_img, note = detect_and_classify(pil_img, force_box)
    c1, c2 = st.columns([1.05, 1])
    with c1:
        st.image(pil_img, caption=caption, use_container_width=True)
        with st.expander('Model crop', expanded=True):
            st.image(crop_img, caption=f'Input region: {note}', width=220)
    with c2:
        st.markdown(
            f"""<div class="result-card">
            <div class="prediction">
            <div class="label">Detected sign</div>
            <div class="sign">{escape(pretty(letter))}</div>
            </div>
            <div class="confidence-track">
            <div class="confidence-fill" style="width:{conf*100:.1f}%"></div>
            </div>
            <div style="color:#64727d;font-weight:700;text-align:center">
            Confidence {conf*100:.1f}%
            </div>
            </div>""",
            unsafe_allow_html=True,
        )
        if st.button('🔊 Speak this letter', key=f'say_{caption}'):
            speak(letter)
        st.markdown('**Top predictions**')
        probability_bars(top5)


def render_asl_text(text, per_row=9, limit=45):
    """Show typed/spoken text as a sequence of ASL reference signs."""
    chars = [c for c in text.upper() if c.isalpha() or c == ' '][:limit]
    if not chars:
        st.info('Nothing to fingerspell.')
        return
    for i in range(0, len(chars), per_row):
        cols = st.columns(per_row)
        for col, ch in zip(cols, chars[i:i + per_row]):
            name = 'space' if ch == ' ' else ch
            path = os.path.join(REFERENCE_DIR, f'{name}.jpg')
            if os.path.exists(path):
                col.image(path, caption=('space' if ch == ' ' else ch),
                          use_container_width=True)
            else:
                col.markdown(f"<div style='text-align:center;font-size:2rem;"
                             f"font-weight:700'>{ch}</div>",
                             unsafe_allow_html=True)


# ==========================================================================
# UI
# ==========================================================================
inject_theme()

pt_files = sorted(set(glob.glob(os.path.join(APP_DIR, '*.pt')) +
                      glob.glob(os.path.join(APP_DIR, 'models', '*.pt'))))
pt_files = [p for p in pt_files if os.path.basename(p) != 'sign_model.pt']
pt_files.sort(key=lambda p: (os.path.basename(p) != 'best_model.pt', p))

with st.sidebar:
    st.header('Model')
    if not pt_files:
        st.error('No `.pt` checkpoint found. Place `best_model.pt` next to app.py.')
        st.stop()
    if len(pt_files) == 1:
        model_path = pt_files[0]
        st.caption(os.path.basename(model_path))
    else:
        model_path = st.selectbox(
            'Checkpoint',
            pt_files,
            format_func=lambda p: os.path.basename(p),
        )

try:
    MODEL, CLASSES, ARCH, IMG_SIZE, META = load_model(model_path)
except Exception as exc:
    st.sidebar.error(f'Failed to load model: {exc}')
    st.stop()
TFM = build_transform(IMG_SIZE)
WORDLIST = load_wordlist(os.path.join(APP_DIR, 'wordlist.txt'))
LETTERS = [c for c in CLASSES if len(c) == 1]   # A–Z, for Practice mode
METRICS = META.get('metrics') or {}
MODEL_NAME = display_model_name(ARCH)

with st.sidebar:
    st.success(MODEL_NAME)
    st.write(f'Classes: **{len(CLASSES)}**')
    st.write(f'Input: **{IMG_SIZE}px**')
    if METRICS:
        st.write(f"Clean val: **{pct(METRICS.get('clean_val_acc'))}**")
        st.write(f"Shifted val: **{pct(METRICS.get('shifted_val_acc'))}**")
    elif META.get('val_acc') is not None:
        st.write(f"Validation: **{META['val_acc']:.4f}**")
    st.write(f'Words: **{len(WORDLIST)}**')
    st.divider()
    st.header('Input')
    if MP_OK:
        st.success('MediaPipe hand tracking')
    else:
        st.info('OpenCV hand detection')
    FORCE_BOX = st.checkbox('Force a fixed centre box instead', value=False,
                            help='Use if detection misbehaves. Restart the '
                                 'camera after changing this.')

st.markdown(
    f"""<div class="hero">
    <h1>The Silent Gap</h1>
    <p>ASL fingerspelling, webcam spelling, video translation, practice,
    speech, and English/Urdu bridging around the improved shift-robust model.</p>
    <div class="hero-badges">
    <span class="badge">{escape(MODEL_NAME)}</span>
    <span class="badge">{len(CLASSES)} classes</span>
    <span class="badge">{IMG_SIZE}px input</span>
    <span class="badge">CPU inference</span>
    </div>
    </div>""",
    unsafe_allow_html=True,
)

metric_cards([
    ('Clean validation', pct(METRICS.get('clean_val_acc')), 'original split'),
    ('Shifted validation', pct(METRICS.get('shifted_val_acc')), 'brightness and scale stress'),
    ('External ASL validation', pct(METRICS.get('external_val_acc')), 'raw Mendeley holdout'),
    ('Autocomplete words', f'{len(WORDLIST):,}', 'live speller dictionary'),
])
dependency_pills()

(tab_live, tab_signs, tab_snap, tab_upload, tab_video, tab_practice,
 tab_bridge, tab_urdu) = st.tabs(
    ['Live Speller', 'Sign Words', 'Snapshot', 'Upload', 'Video',
     'Practice', 'Bridge', 'English ⇄ Urdu'])

# ---------------------------------------------------------------- Live -----
with tab_live:
    section_header('Webcam', 'Live Speller',
                   'Camera input becomes an editable text buffer with speech output.')
    if not WEBRTC_OK:
        empty_state('Webcam unavailable',
                    'Install streamlit-webrtc, av, and opencv-python-headless; Snapshot and Upload still work.')
    else:
        ctx = webrtc_streamer(
            key='asl-live',
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: ASLProcessor(force_box=FORCE_BOX,
                                                         wordlist=WORDLIST),
            media_stream_constraints={'video': True, 'audio': False},
            rtc_configuration={'iceServers': [
                {'urls': ['stun:stun.l.google.com:19302']}]},
            async_processing=True,
        )
        if ctx.video_processor:
            vp = ctx.video_processor
            if not MP_OK:
                state = ('✅ calibrated' if vp.locator.has_background
                         else '— not calibrated')
                st.markdown(f'**Background calibration**  `{state}`')
                bg1, bg2 = st.columns(2)
                if bg1.button('📷 Capture background'):
                    with vp.lock:
                        vp.capture_request = True
                if bg2.button('♻️ Clear background'):
                    with vp.lock:
                        vp.clear_request = True

            st.markdown('**Text controls**')
            b1, b2, b3, b4 = st.columns(4)
            if b1.button('🔊 Speak word'):
                speak(vp.word)
            if b2.button('⌫ Backspace'):
                with vp.lock:
                    vp.word = vp.word[:-1]
            if b3.button('␣ Space'):
                with vp.lock:
                    vp.word += ' '
            if b4.button('🗑️ Clear'):
                with vp.lock:
                    vp.word = ''

            sugg = list(vp.suggestions)
            if sugg:
                st.markdown('**Suggestions**')
                for col, s in zip(st.columns(len(sugg)), sugg):
                    if col.button(f'✓ {s}', key=f'sugg_{s}'):
                        with vp.lock:
                            parts = vp.word.split(' ')
                            vp.word = ' '.join(parts[:-1] + [s]) + ' '

            if st.button('🔊 Speak full sentence'):
                speak(vp.word)
            st.markdown(
                f'<div class="large-output">{escape(vp.word or "Waiting for signs")}</div>',
                unsafe_allow_html=True,
            )

# ----------------------------------------------------------- Sign Words ----
with tab_signs:
    section_header('Dynamic ASL', 'Sign Words',
                   'Word-level sign recognition is enabled when a dynamic model is present.')
    sign_model_path = os.path.join(APP_DIR, 'sign_model.pt')

    if not WEBRTC_OK:
        empty_state('Webcam unavailable',
                    'Install streamlit-webrtc and OpenCV to enable this page.')
    elif not os.path.exists(sign_model_path):
        empty_state('Dynamic model not installed',
                    'Record signs with scripts/collect_signs.py, train notebooks/04_dynamic_signs.ipynb, then place app/sign_model.pt here.')
    elif not MP_OK:
        empty_state('MediaPipe unavailable',
                    'Run on a Python version with a MediaPipe wheel to enable dynamic signs.')
    else:
        sctx = webrtc_streamer(
            key='asl-signs',
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: SignWordProcessor(sign_model_path),
            media_stream_constraints={'video': True, 'audio': False},
            rtc_configuration={'iceServers': [
                {'urls': ['stun:stun.l.google.com:19302']}]},
            async_processing=True,
        )
        if sctx.video_processor:
            svp = sctx.video_processor
            b1, b2 = st.columns(2)
            if b1.button('🔊 Speak sentence'):
                speak(' '.join(s.replace('_', ' ') for s in svp.sentence))
            if b2.button('🗑️ Clear sentence'):
                with svp.lock:
                    svp.sentence = []
            with svp.lock:
                line = ' '.join(s.replace('_', ' ') for s in svp.sentence)
            st.markdown(
                f'<div class="large-output">{escape(line or "Waiting for signs")}</div>',
                unsafe_allow_html=True,
            )

# ------------------------------------------------------------ Snapshot -----
with tab_snap:
    section_header('Camera', 'Snapshot Detection',
                   'Single-frame recognition with the same crop pipeline as live mode.')
    shot = st.camera_input('Take a photo')
    if shot is not None:
        show_result(Image.open(shot), 'captured photo', force_box=FORCE_BOX)
    else:
        empty_state('No snapshot yet', 'Take a photo to see the model crop, prediction, and confidence.')

# -------------------------------------------------------------- Upload -----
with tab_upload:
    section_header('File input', 'Upload Detection',
                   'Classify JPG or PNG hand-sign images with crop inspection.')
    up = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
    if up is not None:
        show_result(Image.open(up), up.name, force_box=FORCE_BOX)
    else:
        empty_state('No image selected', 'Upload a hand-sign image to run detection.')

# -------------------------------------------------------- Translate video --
with tab_video:
    section_header('Video', 'Fingerspelling Translator',
                   'Frame-level predictions are decoded into text.')
    if not WEBRTC_OK:
        empty_state('Video translation unavailable',
                    'Install OpenCV to enable uploaded-video translation.')
    else:
        vid = st.file_uploader('Upload a video',
                               type=['mp4', 'mov', 'avi', 'mkv', 'webm'])
        if vid is not None:
            st.video(vid)
            if st.button('🎬 Translate this video'):
                suffix = os.path.splitext(vid.name)[1] or '.mp4'
                with tempfile.NamedTemporaryFile(delete=False,
                                                 suffix=suffix) as tf:
                    tf.write(vid.getvalue())
                    tmp_path = tf.name
                bar = st.progress(0.0, text='Reading video…')
                try:
                    text, preds = translate_video(
                        tmp_path, classify, force_box=FORCE_BOX,
                        progress=lambda p: bar.progress(
                            p, text='Translating…'))
                finally:
                    bar.empty()
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
                st.session_state['video_result'] = {
                    'name': vid.name, 'text': text, 'preds': preds}

            res = st.session_state.get('video_result')
            if res and res['name'] == vid.name:
                if res['text'].strip():
                    st.markdown('**Translation**')
                    st.markdown(
                        f'<div class="large-output">{escape(res["text"])}</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button('🔊 Speak the translation'):
                        speak(res['text'])
                else:
                    empty_state('No stable signs detected',
                                'Try a clip with a clear hand region and held signs.')
                with st.expander('Frame-by-frame predictions'):
                    st.write('  '.join(pretty(p) for p in res['preds']) or '—')
        else:
            empty_state('No video selected', 'Upload a short fingerspelling video to translate it.')

# ------------------------------------------------------------ Practice -----
with tab_practice:
    section_header('Tutor', 'Practice Mode',
                   'Match the reference sign and track score, accuracy, and streak.')

    ss = st.session_state
    if 'pr_target' not in ss:
        ss.pr_target = random.choice(LETTERS)
        ss.pr_score = ss.pr_attempts = ss.pr_streak = 0
        ss.pr_last_id = None
        ss.pr_last = None
    target = ss.pr_target

    colA, colB = st.columns(2)
    with colA:
        st.markdown(f"### Sign this letter:&nbsp;&nbsp;"
                    f"<span style='font-size:3.4rem;font-weight:800'>"
                    f"{target}</span>", unsafe_allow_html=True)
        ref = os.path.join(REFERENCE_DIR, f'{target}.jpg')
        if os.path.exists(ref):
            st.image(ref, caption=f'reference — how to sign {target}', width=210)
        else:
            st.info('Reference images not found (run make_reference_signs.py).')
    with colB:
        shot = st.camera_input('Show me your sign', key=f'pr_cam_{target}')
        if shot is not None:
            data = shot.getvalue()
            shot_id = hashlib.md5(data).hexdigest()
            if shot_id != ss.pr_last_id:               # score each new photo once
                letter, conf, _, _, _ = detect_and_classify(
                    Image.open(io.BytesIO(data)), force_box=FORCE_BOX)
                correct = (letter == target)
                ss.pr_attempts += 1
                ss.pr_score += int(correct)
                ss.pr_streak = ss.pr_streak + 1 if correct else 0
                ss.pr_last_id = shot_id
                ss.pr_last = (letter, conf, correct)
            letter, conf, correct = ss.pr_last
            if correct:
                st.success(f'✅ Correct! That is **{target}**  ({conf*100:.0f}%)')
            else:
                st.error(f'❌ That looked like **{pretty(letter)}**. '
                         f'Try again, or move on.')
        if st.button('➡️ Next letter'):
            ss.pr_target = random.choice(LETTERS)
            ss.pr_last_id = ss.pr_last = None
            st.rerun()

    acc = (ss.pr_score / ss.pr_attempts * 100) if ss.pr_attempts else 0.0
    metric_cards([
        ('Score', f'{ss.pr_score}/{ss.pr_attempts}', 'correct attempts'),
        ('Accuracy', f'{acc:.0f}%', 'practice session'),
        ('Streak', ss.pr_streak, 'current run'),
        ('Target', target, 'active letter'),
    ])

# ----------------------------------------------------- Communication ------
with tab_bridge:
    section_header('Bridge', 'Speech / Text ⇄ Sign',
                   'Typed or spoken text becomes speech output and ASL reference signs.')

    spoken = None
    if STT_OK:
        st.markdown('**Microphone input**')
        try:
            spoken = speech_to_text(language='en', just_once=True,
                                    use_container_width=True,
                                    start_prompt='🎤 Start speaking',
                                    stop_prompt='⏹️ Stop', key='bridge_stt')
        except Exception:
            spoken = None
    else:
        st.caption('Microphone speech-to-text unavailable. Typed input remains active.')
    if spoken:
        st.session_state['bridge_msg'] = spoken
        st.success(f'🎤 Heard: “{spoken}”')

    typed = st.text_input('…or type a message', key='bridge_typed')
    message = typed.strip() if typed.strip() else \
        st.session_state.get('bridge_msg', '')

    if message:
        st.markdown(
            f'<div class="large-output">{escape(message)}</div>',
            unsafe_allow_html=True,
        )
        if st.button('🔊 Speak it aloud'):       # text/speech -> speech
            speak(message)
        st.markdown('**ASL fingerspelling**')
        render_asl_text(message)                  # text/speech -> sign
    else:
        empty_state('No message yet', 'Type a message to render it as ASL reference signs.')

# ------------------------------------------------------ English <-> Urdu --
with tab_urdu:
    section_header('Translation', 'English ⇄ Urdu',
                   'Typed or spoken input translated with Urdu script rendering.')

    if not TRANSLATE_OK:
        empty_state('Translation unavailable',
                    'Install deep-translator or use the rest of the app offline.')
    else:
        direction = st.radio('Direction',
                             ['English → Urdu', 'Urdu → English'],
                             horizontal=True, key='urdu_dir')
        en_to_ur = direction.startswith('English')
        src, tgt = ('en', 'ur') if en_to_ur else ('ur', 'en')
        src_name = 'English' if en_to_ur else 'Urdu'
        tgt_name = 'Urdu' if en_to_ur else 'English'

        spoken = None
        if STT_OK:
            st.markdown(f'**Speak in {src_name}**')
            try:
                spoken = speech_to_text(
                    language='en-US' if en_to_ur else 'ur-PK',
                    just_once=True, use_container_width=True,
                    start_prompt=f'🎤 Speak {src_name}', stop_prompt='⏹️ Stop',
                    key=f'urdu_stt_{src}')
            except Exception:
                spoken = None
        if spoken:
            st.session_state[f'urdu_input_{src}'] = spoken

        typed = st.text_input(f'…or type in {src_name}',
                              key=f'urdu_typed_{src}')
        text = typed.strip() if typed.strip() else \
            st.session_state.get(f'urdu_input_{src}', '')

        if text:
            try:
                translated = translate_text(text, src, tgt)
            except Exception as exc:
                translated = ''
                st.error(f'Translation service unavailable — try again. ({exc})')
            if translated:
                s_cls = 'large-output urdu' if src == 'ur' else 'large-output'
                s_dir = " dir='rtl'" if src == 'ur' else ''
                st.markdown(
                    f"<div style='color:#64727d;font-size:.9rem;font-weight:700'>{src_name}</div>"
                    f"<div class='{s_cls}'{s_dir} style='font-size:1.35rem'>{escape(text)}</div>",
                    unsafe_allow_html=True)
                st.markdown('')
                t_cls = 'large-output urdu' if tgt == 'ur' else 'large-output'
                t_dir = " dir='rtl'" if tgt == 'ur' else ''
                st.markdown(
                    f"<div style='color:#64727d;font-size:.9rem;font-weight:700'>{tgt_name}</div>"
                    f"<div class='{t_cls}'{t_dir} style='font-size:2.1rem'>{escape(translated)}</div>",
                    unsafe_allow_html=True)
                if st.button(f'🔊 Speak the {tgt_name}'):
                    speak(translated,
                          lang='ur-PK' if tgt == 'ur' else 'en-US')
        else:
            empty_state('No text yet', f'Type in {src_name} to translate.')

st.divider()
st.caption('Recognises A-Z, space, del, and nothing. Built for the Forman CS Club AI Hackathon.')
