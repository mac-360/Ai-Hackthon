# 🤟 The Silent Gap — ASL Sign Language Recognition

**Forman CS Club · AI Hackathon 2026 — Team Berozgar Party**

> Sara is 28 and has been deaf since birth. She signs fluently — but most
> people around her cannot understand her. The technology to close that gap
> exists; it has not been applied with enough focus. This project is a step
> toward closing it.

A system that recognises **American Sign Language** — the 26-letter
fingerspelling alphabet (plus `space`, `del`, `nothing`) from images and live
video — turning it into text and speech. The dynamic word-sign pipeline is
kept as an optional extension path, but no trained dynamic checkpoint is
bundled in the submission.

---

## 🎯 The challenge

Classify a hand sign into **29 classes** (A–Z + `space`, `del`, `nothing`).
Scored on accuracy; ~70k training images, ~17k hidden test images.

## 🔬 What changed

The original benchmark split was clean, uniform 200×200 RGB data. The newer
hidden/contest images are shifted: 300×300, brighter, lower saturation, and
more background-varied. The active model is trained for that shifted setting,
not just the old local validation split.

## 🧠 The model

`app/best_model.pt` is now the improved **shift-robust ConvNeXt-Tiny**
checkpoint. It was fine-tuned on the original labeled images plus the raw
external Mendeley ASL alphabet data, with strong no-flip augmentation:

- random scale/translation/rotation for off-centre hands;
- brightness, gamma, contrast, desaturation, blur, JPEG, and noise;
- no horizontal or vertical flips, because ASL signs are not mirror-invariant.

Validation results for the active checkpoint:

| Split | Accuracy |
|---|---:|
| Clean original validation | 100.00% |
| Shifted local validation | 100.00% |
| External raw ASL validation | 99.76% |

An EfficientNetV2-S companion checkpoint is kept in `models/shift_robust/` for
the final ensemble submission. The old local ResNet, retrieval baseline, and
benchmark-only submissions have been removed.

## 📱 The product — a communication tool, not just a classifier

The `app/` folder is a **Streamlit app** that turns the model into something
usable — **7 bundled tabs**:

| Tab | What it does |
|---|---|
| 📹 **Live Speller** | Real-time webcam — detects the hand, recognises fingerspelling, builds words with **predictive autocomplete**, speaks them aloud. |
| 📸 **Snapshot** | Take one photo → instant letter. |
| 🖼️ **Upload** | Classify any hand-sign image. |
| 🎬 **Translate Video** | Upload a fingerspelling video → it reads it frame by frame and translates it into text. |
| 🎓 **Practice** | Learn the alphabet — shown a letter + reference, you sign it, it scores you. |
| 🔁 **Communication Bridge** | Speech / text → ASL fingerspelling + Kokoro/browser speech — bridges all three modalities, both directions. |
| 🌐 **English ⇄ Urdu** | Type or speak → translate between English and Urdu in proper script, both ways, with speech. |

If a trained `app/sign_model.pt` dynamic-sign checkpoint is added later, the
app reveals an extra **Sign Words** tab for MediaPipe-Holistic + LSTM
word-sign recognition.

It **detects the hand** before classifying — **MediaPipe** where available,
otherwise OpenCV skin-segmentation with optional **background calibration**
(detects the hand by what changed, so it works even against a same-coloured
wall).

English speech uses offline Kokoro TTS when `app/kokoro_models/kokoro-v1.0.onnx`
and `app/kokoro_models/voices-v1.0.bin` are present. The app falls back to the
browser Web Speech API when Kokoro is unavailable or the language is not
supported.

## 📂 Repository

```
.
├── app/                       # the Streamlit app — self-contained & deployable
│   ├── app.py                 #   app: spell · video · practice · bridge · Urdu …
│   ├── hand_detection.py      #   MediaPipe / skin-seg / background-subtraction
│   ├── video_translate.py     #   fingerspelling-video → text
│   ├── sign_recognition.py    #   dynamic word-level sign LSTM + live recogniser
│   ├── best_model.pt          #   improved shift-robust ConvNeXt-Tiny checkpoint
│   ├── sign_model.pt          #   optional dynamic-sign LSTM, not bundled
│   ├── reference_signs/       #   29 reference images (Practice mode)
│   ├── wordlist.txt           #   autocomplete dictionary
│   ├── kokoro_models/         #   optional local Kokoro ONNX + voices
│   └── requirements.txt
├── notebooks/                 # dynamic-sign notebook only
│   └── 04_dynamic_signs.ipynb
├── scripts/                   # robust training, inference, and tests
├── submissions/               # final robust submission CSV
├── models/shift_robust/       # improved ensemble checkpoint + training logs
├── deploy/                    # Hugging Face Spaces deployment package
└── data/                      # competition dataset (not committed)
```

## ▶️ Run it

**The app**
```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

**Train / infer the robust model**
```bash
python scripts/train_shift_robust.py \
  --external-root 'data/external/mendeley_root/Root/Type_01_(Raw_Gesture)' \
  --models convnext_tiny tf_efficientnetv2_s \
  --epochs 4 --steps-per-epoch 320 --batch-size 256 --img-size 256

python scripts/balance_az_submission.py
```

The final contest upload candidate is `submissions/submission.csv`.

**Deploy** — see [`deploy/DEPLOY.md`](deploy/DEPLOY.md) for a public
Hugging Face Space.

## ⚠️ Limitations (and we mean it)

- **Vocabulary.** The core model covers the 26-letter alphabet. Real ASL is a
  full language — thousands of signs, facial grammar, context — so this remains
  a slice of it.
- **Dataset bias.** The alphabet training images are one hand, one
  background, and even lighting. The improved model is stress-tested beyond
  that setting, but it still inherits those assumptions.
- **Real-world gap.** The active model is more robust to lighting, background,
  and hand-scale shift, but hand-cropping still matters for webcam use.

## 🔭 Future work

1. **Train and bundle the dynamic vocabulary** — the MediaPipe-Holistic + LSTM
   code path is present, but it needs a real `app/sign_model.pt` checkpoint
   from more signers and more signs before it should be shown in the demo.
2. **Continuous signing** — segment a stream of signs into sentences, rather
   than recognising one isolated sign at a time.
3. **Diverse data** — many hands, skin tones, backgrounds and lighting so the
   models generalise beyond the benchmark.

## 🙌 Team

**Berozgar Party** — Forman Computer Science Club AI Hackathon 2026.

*AI has the raw capability to close the silent gap. This is one honest step.*
