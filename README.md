# The Silent Gap - Streamlit Submission

This folder is a standalone submission package for the improved ASL
Streamlit app and model artifacts.

## Contents

- `app/` - runnable Streamlit app, active `best_model.pt`, reference signs,
  word list, and app requirements.
- `models/shift_robust/` - improved ConvNeXt-Tiny and EfficientNetV2-S
  checkpoints with training logs/manifests.
- `submissions/submission.csv` - final generated contest submission CSV.
- `scripts/` - lightweight verification and inference helpers.
- `deploy/` - Hugging Face Spaces deployment notes and system packages.
- `PROJECT_README.md` - full project writeup copied from the working project.

## Run Locally

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

The app defaults to OpenCV hand detection because some hosted notebook
runtimes crash during native MediaPipe import. On a local machine where
MediaPipe is stable, run:

```bash
ASL_ENABLE_MEDIAPIPE=1 streamlit run app/app.py
```

## Verify

```bash
python -m py_compile app/app.py app/hand_detection.py app/video_translate.py app/sign_recognition.py
python scripts/verify_app.py
python scripts/test_handdetect.py
python scripts/test_video.py
```

The verification scripts skip data-dependent checks when the original training
dataset is not present.

## Git Notes

Model checkpoints are tracked with Git LFS via `.gitattributes`. Run
`git lfs install` before pushing this repository to a remote.
