# yt-dashboard (prepared export)

This repository contains a Streamlit dashboard exported from the user's Replit project.

Files included:
- app.py (Streamlit app)
- collector.py, youtube_api.py, database.py, utils.py (supporting modules)
- requirements.txt (generated from pyproject/uv.lock and augmented)

How to deploy to Streamlit Cloud:
1) Create a new GitHub repository (e.g. `yt-dashboard`).
2) Upload all files from this archive to the repo root.
3) In Streamlit Cloud -> New app -> choose `youruser/yt-dashboard`, branch `main`, main file `app.py`.
4) Deploy. If the repo is private, either make it public or connect GitHub in Streamlit account settings.

If you want me to automatically create a GitHub repo and push these files, I can provide step-by-step git commands you can run locally.

requirements.txt content generated:
```
streamlit==1.42.2
pandas==2.2.3
plotly==6.0.0
matplotlib
numpy==2.2.3
google-api-python-client
google-auth
google-auth-oauthlib
oauth2client
openpyxl>=3.1.5
pandas>=2.2.3
plotly>=6.0.0
psycopg2-binary>=2.9.10
requests>=2.32.3
streamlit>=1.42.2
trafilatura>=2.0.0
twilio>=9.4.6
```