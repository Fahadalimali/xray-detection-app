"""Standalone Streamlit X-ray detector for Streamlit Community Cloud."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


MODEL_PATH = Path(__file__).resolve().parents[1] / "best.pt"


@st.cache_resource(show_spinner="Loading detection model...")
def load_model() -> YOLO:
    """Load weights once per Streamlit server process."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "best.pt was not found. Add the model file to the repository root and redeploy."
        )
    return YOLO(str(MODEL_PATH))


st.set_page_config(page_title="X-ray Detector", page_icon="🩻", layout="wide")
st.title("🩻 X-ray Detector")
st.caption("Upload an X-ray image to detect objects with the trained YOLO model.")

with st.sidebar:
    st.header("Settings")
    confidence = st.slider("Minimum confidence", 0.05, 0.95, 0.25, 0.05)
    st.caption("The model runs directly in Streamlit Cloud. No separate API is needed.")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png", "webp"])

if not uploaded_file:
    st.info("Upload a JPG, PNG, or WEBP image to get started.")
    st.stop()

try:
    image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
except (UnidentifiedImageError, OSError):
    st.error("Please upload a valid image file.")
    st.stop()

left, right = st.columns(2)
with left:
    st.subheader("Original")
    st.image(image, use_container_width=True)

if st.button("Analyze image", type="primary"):
    try:
        with st.spinner("Running detection..."):
            result = load_model().predict(image, conf=confidence, verbose=False)[0]
            # Ultralytics returns an OpenCV-style BGR image from plot().
            annotated = Image.fromarray(result.plot()[:, :, ::-1])
    except Exception as exc:
        st.error(f"Unable to run the model: {exc}")
        st.stop()

    rows = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            x1, y1, x2, y2 = (round(value, 1) for value in box.xyxy[0].tolist())
            rows.append(
                {
                    "Label": str(result.names[class_id]),
                    "Confidence": f"{float(box.conf[0].item()):.1%}",
                    "Bounding box": f"({x1}, {y1}) → ({x2}, {y2})",
                }
            )

    with right:
        st.subheader("Detections")
        st.image(annotated, use_container_width=True)

    st.metric("Objects detected", len(rows))
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No detections found at the selected confidence threshold.")
