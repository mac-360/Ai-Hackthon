# The Silent Gap — Streamlit App

Polished demo app for the improved shift-robust ASL fingerspelling model in
`app/best_model.pt`.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

From the repo root:

```bash
streamlit run app/app.py
```

## Features

| Tab | Capability |
|---|---|
| Live Speller | Webcam ASL fingerspelling to editable text and speech |
| Snapshot | Single camera photo classification |
| Upload | Image upload classification with model crop and top predictions |
| Video | Uploaded fingerspelling video to decoded text |
| Practice | Reference sign, camera attempt, score, accuracy, streak |
| Bridge | Typed or spoken English text to Kokoro/browser speech and ASL reference signs |
| English ⇄ Urdu | English/Urdu translation with Urdu script rendering |

The optional Sign Words tab is hidden unless a trained `app/sign_model.pt`
dynamic-sign checkpoint is present.

## Offline Kokoro TTS

The app uses Kokoro TTS for English speech when `kokoro-tts` and these local
model files are available:

```text
app/kokoro_models/kokoro-v1.0.onnx
app/kokoro_models/voices-v1.0.bin
```

Download them from the Kokoro release:

```bash
mkdir -p app/kokoro_models
curl -L -o app/kokoro_models/kokoro-v1.0.onnx https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx
curl -L -o app/kokoro_models/voices-v1.0.bin https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin
```

On unsupported Python versions or for unsupported languages, the app falls
back to the browser Web Speech API.

## Fallbacks

- If MediaPipe is unavailable, the app uses OpenCV hand detection.
- MediaPipe is opt-in because some hosted notebook runtimes crash during its
  native import. Start with `ASL_ENABLE_MEDIAPIPE=1 streamlit run app.py` if
  you want to use it locally.
- If webcam packages are unavailable, Snapshot and Upload remain usable.
- If microphone capture is unavailable, typed input remains usable.
- If translation is unavailable, all ASL recognition features still work.
- If Kokoro TTS is unavailable, the app uses browser speech output.

The app reads the checkpoint class list, architecture, input size, and metrics
from `best_model.pt`, so retrained compatible checkpoints can be dropped in
without changing the UI.
