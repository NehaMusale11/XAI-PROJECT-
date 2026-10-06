# 🚀 Deployment Guide: Post-hoc Explanation Web Application

This document provides complete instructions for deploying the **Explainable AI (XAI) Post-hoc Explanation Dashboard** to free and production cloud hosting platforms.

---

## 📌 Summary of Deployment Configurations Included

The repository includes production-ready deployment configurations:
- **`Procfile`**: For **Render**, **Railway**, and **Heroku** WSGI runners (`gunicorn` with worker/thread tuning & 180s timeout).
- **`render.yaml`**: Infrastructure-as-Code Blueprint for **1-Click Render.com** deployment.
- **`Dockerfile` & `.dockerignore`**: Universal container configuration for **Docker**, **AWS App Runner**, **GCP Cloud Run**, **Fly.io**, and **Hugging Face Spaces**.
- **`docker-compose.yml`**: One-command local or VPS container deployment.
- **`runtime.txt`**: Specifies `python-3.11.6` for automated PaaS platform detection.
- **`/healthz` & `/api/health`**: Zero-downtime healthcheck probes for cloud load balancers.

---

## 🌐 Option 1: Render.com (Recommended — 100% Free Tier)

Render provides free hosting for web services connected directly to GitHub.

### Steps to Deploy:
1. Log in to [Render.com](https://render.com/).
2. Click **New +** in the top navigation and select **Web Service**.
3. Choose **Build and deploy from a Git repository**.
4. Connect your GitHub account and select this repository: `NehaMusale11/XAI-PROJECT-`.
5. Configure the service settings (or Render will automatically detect `render.yaml` / `Procfile`):
   - **Name**: `xai-explanation-dashboard`
   - **Language / Runtime**: `Python 3`
   - **Region**: `Oregon (US West)` or closest to you
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --workers 1 --threads 4 --timeout 180`
   - **Instance Type**: `Free`
6. Click **Create Web Service**.
7. Render will build the environment, install the packages, and launch your dashboard at:
   `https://<your-service-name>.onrender.com`

---

## 🤗 Option 2: Hugging Face Spaces (Free with 16 GB RAM)

Hugging Face Spaces is an ideal platform for machine learning web apps.

### Steps to Deploy:
1. Create a free account on [Hugging Face](https://huggingface.co/).
2. Go to **Spaces** -> **Create new Space**.
3. Choose:
   - **Space Name**: `xai-explanation-dashboard`
   - **License**: `mit` or `openrail`
   - **Space SDK**: **Docker** -> **Blank**
   - **Space hardware**: `CPU Basic (free - 2 vCPU, 16 GB RAM)`
4. Hugging Face will provide a git repository URL for the space:
   `https://huggingface.co/spaces/<username>/xai-explanation-dashboard`
5. Push this codebase to the Hugging Face space remote:
   ```bash
   git remote add hf https://huggingface.co/spaces/<username>/xai-explanation-dashboard
   git push -u hf main
   ```
6. Hugging Face automatically builds the `Dockerfile` and presents your dashboard at a permanent public URL.

---

## 🚂 Option 3: Railway.app

1. Go to [Railway.app](https://railway.app/).
2. Click **Start a New Project** -> **Deploy from GitHub repo**.
3. Select `NehaMusale11/XAI-PROJECT-`.
4. Railway will automatically detect the `Dockerfile` or `Procfile`.
5. Under service settings, generate a domain (e.g. `xai-dashboard.up.railway.app`).

---

## 🐳 Option 4: Docker (Self-Hosted / VPS / Local Container)

You can build and run the production container anywhere with Docker installed:

### Build and Run with Docker:
```bash
# Build the container image
docker build -t xai-dashboard:latest .

# Run the container exposing port 5000
docker run -d -p 5000:5000 --name xai_app xai-dashboard:latest
```

### Or with Docker Compose:
```bash
docker compose up -d
```
Access the application at: `http://localhost:5000`

---

## 💻 Option 5: Local Production Serving (Cross-Platform)

To run the application locally with a production WSGI server instead of development mode:

### On Windows (using Waitress):
```bash
waitress-serve --listen=0.0.0.0:5000 app:app
```

### On Linux / macOS (using Gunicorn):
```bash
gunicorn app:app --workers 1 --threads 4 --timeout 180 --bind 0.0.0.0:5000
```
