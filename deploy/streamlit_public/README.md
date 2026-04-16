# Public Deployment

This folder contains a lightweight Streamlit app intended for public hosting.

## Why this deployment app exists

The training project uses heavier ML dependencies such as `scikit-surprise`. Public hosts like Streamlit Community Cloud are usually more reliable when the serving app uses a smaller dependency set. This app serves precomputed Bollywood recommendations from `bollywood_public_bundle.joblib`.

## Quick Start (Local)

```bash
# From repo root
python3 -m venv .venv
source .venv/bin/activate
pip install -r deploy/streamlit_public/requirements.txt

# Generate demo data (if bundle doesn't exist yet)
pip install numpy
python deploy/streamlit_public/generate_demo_bundle.py

# Run the app
streamlit run deploy/streamlit_public/app.py
```

## Build the production bundle

If you have the full training pipeline and models available:

```bash
.venv/bin/python deploy/streamlit_public/build_public_bundle.py
```

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
3. Click **New app** and select this repo.
4. Set the entrypoint to: `deploy/streamlit_public/app.py`
5. Streamlit will use the `requirements.txt` in this folder automatically.
6. Click **Deploy** — the app goes live in minutes.

## Deploy to Render

1. Push this repository to GitHub.
2. Go to [render.com](https://render.com) and create a new **Blueprint**.
3. Connect the repo — Render detects `render.yaml` automatically.
4. Click **Apply** — the web service starts the Streamlit app.
