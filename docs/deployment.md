# Deployment

The project supports three deployment shapes. Pick whichever matches
your needs.

## 1. Local — `make docker`

Brings up the full stack (FastAPI on :8000 + Streamlit on :8501) via
docker-compose. The Streamlit container runs in API mode (talks to the
api container via `CHURN_API_URL=http://api:8000`).

```bash
git clone https://github.com/MeetLunagariya/churn-prediction.git
cd churn-prediction
make docker      # build + start both services
```

## 2. Streamlit Community Cloud — public demo URL

Deploys only the Streamlit dashboard. It runs in **embedded mode** (no
separate API), calling the in-process `ChurnPredictor` directly. The
committed model artifact (`models/churn_v1.joblib`) means the cold
start is fast — no training at boot time.

**Steps:**

1. Go to https://share.streamlit.io and sign in with GitHub (same
   account that owns the repo).
2. Click **New app** → pick the `MeetLunagariya/churn-prediction` repo.
3. Configure:
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
   - **Python version:** 3.11
   - (Optional) **App URL:** custom subdomain like
     `churn-meetlunagariya.streamlit.app`
4. Click **Deploy**. First build takes ~3 minutes (pip installs all
   deps). Subsequent updates are pushed automatically on every commit
   to `main`.

The dashboard fetches the dataset on first run (the download is small,
~1 MB) and reuses it across requests via Streamlit's resource cache.

## 3. Render or Fly.io — FastAPI as a public API

Deploys the API service. Useful if you want to point external clients
or an external Streamlit instance at a live `/predict` URL.

### Render (browser-only setup)

1. Sign in to https://render.com with GitHub.
2. **New** → **Web Service** → connect the repo.
3. Configure:
   - **Environment:** Docker
   - **Dockerfile path:** `Dockerfile`
   - **Branch:** `main`
   - **Region:** closest to you
   - **Instance type:** Free
4. Render auto-detects the Dockerfile and builds the image. The HEALTHCHECK
   hits `/health`; once it's green, your API is live at
   `https://<service-name>.onrender.com`.

Caveats: the Render free tier sleeps after ~15 min of inactivity, so the
first request after a quiet period takes ~30s while the container wakes.

### Fly.io (CLI setup)

```bash
brew install flyctl   # or curl -L https://fly.io/install.sh | sh
flyctl auth login
flyctl launch         # answers most questions from the Dockerfile
flyctl deploy
```

Better cold-start latency than Render free, no built-in sleep.

## Pointing the deployed dashboard at the deployed API

If you deploy both (Streamlit Cloud + Render/Fly), set the Streamlit
Cloud secret `CHURN_API_URL=https://<your-api>.onrender.com` (or the
Fly URL). The dashboard switches to API mode automatically — see the
mode banner at the top of the page.
