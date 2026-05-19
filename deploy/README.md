---
title: ASL Sign Detector
emoji: 🤟
colorFrom: indigo
colorTo: green
sdk: streamlit
app_file: app.py
python_version: "3.11"
pinned: false
license: mit
---

# 🤟 The Silent Gap — ASL Sign Detector & Speller

Real-time American Sign Language fingerspelling recognition.

- **📹 Live Speller** — sign with your webcam; letters are detected in real
  time and spelled into words you can hear spoken aloud.
- **📸 Snapshot** — take one photo, get the letter instantly.
- **🖼️ Upload** — classify any hand-sign image.

Recognises 29 classes (A–Z, space, del, nothing). Built with a fine-tuned
ConvNeXt model for the **Forman CS Club AI Hackathon 2026** by team
**Berozgar Party**.

> On Python 3.11 the Space uses **MediaPipe** for automatic hand detection.
> If the live webcam has trouble connecting, the **Snapshot** tab always works.
