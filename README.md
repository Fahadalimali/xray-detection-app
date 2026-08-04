# X-ray Detection App

The project provides a FastAPI inference service and a Streamlit upload interface for `best.pt`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

Start the API in one terminal:

```powershell
uvicorn backend.main:app --reload
```

Start the web interface in another terminal:

```powershell
streamlit run frontend/app.py
```

Open `http://localhost:8501`, upload an image, and choose **Analyze image**. The interactive API documentation is available at `http://127.0.0.1:8000/docs`.

Set `MODEL_PATH` to use weights from a different location, or set `API_URL` before launching Streamlit to connect to another API server.
