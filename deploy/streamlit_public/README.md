# Public Deployment

This folder contains a lightweight Streamlit app intended for public hosting.

## Why this deployment app exists

The training project uses heavier ML dependencies such as `scikit-surprise`. Public hosts like Streamlit Community Cloud are usually more reliable when the serving app uses a smaller dependency set. This app serves precomputed Bollywood recommendations from `bollywood_public_bundle.joblib`.

## Build the public bundle

From the project root:

```bash
.venv/bin/python deploy/streamlit_public/build_public_bundle.py
```

## Streamlit Community Cloud

- Push this repository to GitHub.
- In Streamlit Community Cloud, choose the entrypoint:
  `deploy/streamlit_public/app.py`
- Streamlit will use the `requirements.txt` file in this folder.
- The included `bollywood_public_bundle.joblib` is already filtered to the Bollywood-focused TIMDB catalog, so no training step is required at deploy time.

## Render

- Push this repository to GitHub.
- Create a new Blueprint on Render from the repo.
- Render will detect `render.yaml` in the repository root.
- The web service starts the Streamlit app directly from `deploy/streamlit_public/app.py`.
