# 🚀 Deploying the ASL Sign Detector

Two free options. **Hugging Face Spaces** is recommended (it supports the
`packages.txt` system libraries MediaPipe/OpenCV need).

---

## Option A — Hugging Face Spaces (recommended)

The Hugging Face Space is a flat folder — copy these items into its root.

### What goes in the Space
| File | From |
|---|---|
| `app.py` | `app/` |
| `hand_detection.py` | `app/` |
| `video_translate.py` | `app/` |
| `sign_recognition.py` | `app/` |
| `wordlist.txt` | `app/` |
| `reference_signs/` | `app/` (29 images for Practice mode) |
| `best_model.pt` | `app/` (≈111 MB → uses Git LFS) |
| `sign_model.pt` | `app/` — *optional*; include it if the dynamic-sign model has been trained (Sign Words tab) |
| `requirements.txt` | `app/` (one file for everywhere — MediaPipe auto-installs on the Space's Python 3.11) |
| `packages.txt` | `deploy/` |
| `README.md` | `deploy/` (has the Spaces config header) |
| `.gitattributes` | `deploy/` (sets up LFS for `*.pt`) |

### Steps

1. Create a free account at **https://huggingface.co**.
2. **New → Space**. Name it (e.g. `asl-sign-detector`), **SDK: Streamlit**,
   hardware **CPU basic** (free). Create.
3. Easiest path — the **web UI**: open the Space → **Files → Add file →
   Upload files**, and drop in all nine items above. The browser uploader
   sends `best_model.pt` as an LFS file automatically. Commit.
4. The Space builds (a few minutes) and goes live at
   `https://huggingface.co/spaces/<you>/asl-sign-detector`.

### Or via git (for larger/repeat uploads)

```bash
git clone https://huggingface.co/spaces/<you>/asl-sign-detector
cd asl-sign-detector

# copy the files in (from the project's app/ and deploy/ folders):
cp /path/to/app/{app.py,hand_detection.py,video_translate.py,sign_recognition.py,wordlist.txt,best_model.pt,requirements.txt} .
cp /path/to/app/sign_model.pt .          # optional — only if trained
cp -r /path/to/app/reference_signs .
cp /path/to/deploy/{packages.txt,README.md,.gitattributes} .

git lfs install
git add .gitattributes
git add app.py hand_detection.py video_translate.py wordlist.txt reference_signs requirements.txt packages.txt README.md
git add best_model.pt              # tracked by LFS via .gitattributes
git commit -m "ASL Sign Detector"
git push
```

> The `python_version: "3.11"` line in `README.md` is important — it lets
> **MediaPipe** install, so the deployed Space gets automatic hand detection.

---

## Option B — Streamlit Community Cloud

1. Push the project to a **public GitHub repo** (commit `best_model.pt`
   with Git LFS).
2. Go to **https://share.streamlit.io** → **New app** → pick the repo,
   set the main file to **`app/app.py`**.
3. Deploy.

Note: Streamlit Cloud has no `packages.txt` equivalent, so MediaPipe may not
install — the app then falls back to the **guide-box** mode automatically
(still fully functional).

---

## After deploying

- The **Snapshot** and **Upload** tabs work everywhere.
- The **Live Speller** uses WebRTC. On a cloud host, browser↔server video may
  need a TURN server; if the live feed will not connect, the Snapshot tab is
  the reliable demo path. It works flawlessly when running locally
  (`streamlit run app/app.py`).
- Share the public URL with the judges — a working link beats a laptop demo.
