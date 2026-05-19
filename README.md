# The Silent Gap

The Silent Gap is a Streamlit-based ASL fingerspelling assistant built for the
Forman CS Club AI Hackathon 2026. It recognizes American Sign Language alphabet
signs from webcam, image, and video input, then turns those predictions into
usable communication workflows: live spelling, speech output, practice scoring,
ASL reference rendering, and English <-> Urdu translation.

The bundled model classifies 29 signs: `A-Z`, `space`, `del`, and `nothing`.
The repository is packaged as a complete submission with the runnable app,
active checkpoint, reference assets, validation scripts, deployment files, and
the final contest submission CSV.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Model Details](#model-details)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Verification](#verification)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Team](#team)

## Project Overview

Most ASL alphabet classifiers stop at single-image prediction. This project
wraps the classifier in a complete communication tool:

- real-time webcam spelling with debounced predictions;
- single-image and uploaded-image classification;
- uploaded-video decoding for held fingerspelling signs;
- guided practice with reference signs and scoring;
- speech and text bridging for hearing and deaf users;
- English <-> Urdu translation support;
- optional dynamic word-sign recognition when an LSTM checkpoint is supplied.

The app is designed to run on CPU and degrade gracefully when optional packages
are unavailable. For example, if MediaPipe cannot be imported, the hand locator
falls back to OpenCV-based detection instead of crashing.

## Key Features

| Feature | Description | Main files |
| --- | --- | --- |
| Live Speller | Reads webcam frames, detects the hand, classifies stable signs, builds an editable text buffer, suggests word completions, and speaks the result. | `app/app.py`, `app/hand_detection.py` |
| Snapshot Detection | Captures a single camera photo and displays the predicted sign, confidence, top predictions, and model crop. | `app/app.py` |
| Upload Detection | Classifies uploaded JPG/PNG images through the same detection and inference pipeline. | `app/app.py` |
| Video Translation | Samples uploaded videos, classifies frame-level signs, and decodes stable prediction runs into text. | `app/video_translate.py` |
| Practice Mode | Shows ASL reference signs, checks camera attempts, and tracks score, accuracy, streak, and target letter. | `app/app.py`, `app/reference_signs/` |
| Communication Bridge | Converts typed or spoken text into speech output and ASL reference-sign rendering. | `app/app.py`, `app/wordlist.txt` |
| English <-> Urdu | Translates typed or spoken English/Urdu input and renders Urdu text correctly. | `app/app.py` |
| Optional Sign Words | Enables word-level dynamic sign recognition when `app/sign_model.pt` is present. | `app/sign_recognition.py` |

## System Architecture

```text
Camera / Image / Video
        |
        v
Hand localization
  - MediaPipe Hands when explicitly enabled
  - OpenCV background subtraction or skin segmentation fallback
  - Fixed center box fallback when needed
        |
        v
Preprocessing
  - square crop
  - resize to checkpoint input size
  - ImageNet normalization
        |
        v
ASL classifier
  - ConvNeXt-Tiny via timm
  - 29 output classes
        |
        v
Application workflows
  - live text buffer
  - snapshot/upload result card
  - video sequence decoder
  - practice scoring
  - speech and translation views
```

### Recognition Flow

The classifier is never called blindly on full webcam frames unless detection
fails. `HandLocator` first selects a likely hand region, and the selected crop
is passed through the model. The UI exposes the crop so users can see what the
model actually received.

For live spelling, the app keeps a rolling window of recent labels. A letter is
committed only after repeated agreement and sufficient confidence. This avoids
most accidental text updates caused by hand movement, blur, or intermediate
gestures.

For video translation, frame predictions are collapsed with `decode_sequence`.
Only labels that persist for a minimum run length and confidence are emitted;
short runs are treated as noise.

## Model Details

`app/best_model.pt` is the active checkpoint used by the Streamlit app.

| Property | Value |
| --- | --- |
| Architecture | ConvNeXt-Tiny |
| Framework | PyTorch + timm |
| Classes | 29 (`A-Z`, `space`, `del`, `nothing`) |
| Input size | 256 x 256 |
| Epochs | 4 |
| Clean validation accuracy | 100.00% |
| Shifted validation accuracy | 100.00% |
| External ASL validation accuracy | 99.76% |

The active model was trained for robustness against distribution shift in the
hidden contest set. Training used the original labeled alphabet data plus raw
external ASL alphabet images, with augmentation focused on off-center hands,
brightness changes, contrast changes, blur, JPEG artifacts, noise, and
desaturation. Horizontal and vertical flips were avoided because ASL signs are
not mirror-invariant.

Additional training logs, manifests, and companion checkpoints are stored in
`models/shift_robust/`.

## Repository Structure

```text
.
|-- app/
|   |-- app.py                  # Streamlit entry point and user interface
|   |-- hand_detection.py       # MediaPipe/OpenCV/fixed-box hand localization
|   |-- video_translate.py      # Uploaded-video frame sampling and decoding
|   |-- sign_recognition.py     # Optional MediaPipe Holistic + LSTM pipeline
|   |-- best_model.pt           # Active ASL fingerspelling checkpoint
|   |-- reference_signs/        # 29 reference images for Practice and Bridge
|   |-- wordlist.txt            # Autocomplete word list
|   |-- requirements.txt        # App dependencies
|   `-- README.md               # App-specific run notes
|-- models/shift_robust/
|   |-- best_convnext_tiny.pt
|   |-- best_tf_efficientnetv2_s.pt
|   |-- train_config.json
|   `-- history_*.csv
|-- scripts/
|   |-- verify_app.py           # Model-loading and classification check
|   |-- test_handdetect.py      # Hand-localization checks
|   |-- test_video.py           # Video decoder and synthetic clip check
|   `-- infer_shift_robust.py   # Ensemble inference helper
|-- submissions/
|   `-- submission.csv          # Final generated contest submission
|-- deploy/
|   |-- DEPLOY.md               # Hosting guide
|   |-- README.md               # Hugging Face Spaces README
|   `-- packages.txt            # System packages for hosted deployment
|-- PROJECT_README.md           # Original long-form project write-up
`-- README.md                   # Project README
```

## Getting Started

### Prerequisites

- Python 3.11 is recommended for full MediaPipe support.
- Python 3.13 can run the app with OpenCV fallback detection.
- Git LFS is required if pushing checkpoints to a remote repository.
- A webcam is required for live spelling, snapshot capture, practice mode,
  microphone input, and optional dynamic-sign recognition.

### Install

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r app\requirements.txt
```

### Run

```powershell
streamlit run app\app.py
```

The app will start with OpenCV hand detection by default. To use MediaPipe
where it is installed and stable:

```powershell
$env:ASL_ENABLE_MEDIAPIPE = "1"
streamlit run app\app.py
```

## Configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `ASL_ENABLE_MEDIAPIPE` | Enables lazy MediaPipe import for hand tracking and optional dynamic-sign recognition. | Disabled |
| `ASL_DISABLE_WEBRTC` | Disables live webcam/WebRTC paths for constrained environments or tests. | Disabled |
| `ASL_KOKORO_DIR` | Directory containing optional Kokoro model files. | `app/kokoro_models` |
| `ASL_KOKORO_MODEL` | Path to the Kokoro ONNX model. | `app/kokoro_models/kokoro-v1.0.onnx` |
| `ASL_KOKORO_VOICES` | Path to the Kokoro voice file. | `app/kokoro_models/voices-v1.0.bin` |
| `ASL_KOKORO_VOICE` | Kokoro voice ID for English speech. | `af_sarah` |
| `ASL_KOKORO_SPEED` | Kokoro speech speed, clamped between 0.75 and 1.25. | `0.95` |

### Optional Offline Text-to-Speech

English speech uses Kokoro TTS when the dependency and local model files exist:

```text
app/kokoro_models/kokoro-v1.0.onnx
app/kokoro_models/voices-v1.0.bin
```

If Kokoro is unavailable, unsupported by the Python version, or missing model
files, the app falls back to the browser Web Speech API.

### Optional Dynamic Sign Model

The Sign Words tab is hidden unless this file exists:

```text
app/sign_model.pt
```

That checkpoint must match the `SignLSTM` format in `app/sign_recognition.py`.
Without it, the app remains a fingerspelling-focused system.

## Verification

Run the lightweight checks from the repository root:

```powershell
python -m py_compile app\app.py app\hand_detection.py app\video_translate.py app\sign_recognition.py
python scripts\verify_app.py
python scripts\test_handdetect.py
python scripts\test_video.py
```

`verify_app.py`, `test_handdetect.py`, and `test_video.py` skip
dataset-dependent checks when the original training data is not present.

## Deployment

Hugging Face Spaces is the recommended hosting target because it supports the
system packages required by OpenCV and MediaPipe. Full deployment instructions
are in `deploy/DEPLOY.md`.

For a Hugging Face Space, copy the app files from `app/`, the `reference_signs/`
directory, `best_model.pt`, `requirements.txt`, `deploy/packages.txt`, and the
Spaces-specific `deploy/README.md` into the Space root. Model checkpoints are
tracked with Git LFS through `.gitattributes`.

## Troubleshooting

| Issue | Likely cause | Action |
| --- | --- | --- |
| MediaPipe does not load | Unsupported Python version or native package issue | Run without `ASL_ENABLE_MEDIAPIPE`; the app will use OpenCV fallback detection. |
| Webcam tab is unavailable | Missing `streamlit-webrtc`, `av`, or OpenCV dependency | Reinstall from `app/requirements.txt`; Snapshot and Upload can still work without webcam streaming. |
| Live spelling adds wrong letters | Hand crop is unstable or confidence is low | Improve lighting/background, use background calibration, or enable the fixed center box from the sidebar. |
| Kokoro speech is unavailable | Missing model files or unsupported Python version | Add Kokoro files under `app/kokoro_models` or use the browser speech fallback. |
| Translation fails | `deep-translator` is missing or provider/network access is unavailable | Reinstall dependencies and check network access; ASL recognition remains usable. |

## Limitations

- The bundled checkpoint recognizes ASL alphabet fingerspelling, not complete
  ASL grammar, facial grammar, or continuous natural signing.
- Real-world accuracy depends on crop quality, camera resolution, lighting,
  hand position, motion blur, and background contrast.
- Video translation assumes signs are held long enough to create stable frame
  runs.
- The optional dynamic-sign path requires a separately trained
  `app/sign_model.pt` checkpoint before it appears in the UI.
- English <-> Urdu translation depends on `deep-translator` and external
  translation service availability.

## Team

Built for the Forman CS Club AI Hackathon 2026 by Team Berozgar Party.
