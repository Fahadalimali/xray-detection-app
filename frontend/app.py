"""Streamlit interface for the X-ray Detection API."""

from __future__ import annotations

import io
import os
from typing import Any

import requests
import streamlit as st
from PIL import Image, ImageDraw


DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def draw_detections(image: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
    """Draw returned API boxes directly onto a copy of the uploaded image."""
    annotated = image.convert("RGB").copy()
    canvas = ImageDraw.Draw(annotated)
    line_width = max(2, image.width // 250)
    for detection in detections:
        box = detection["box"]
        coords = (box["x1"], box["y1"], box["x2"], box["y2"])
        label = f'{detection["label"]} {detection["confidence"]:.0%}'
        canvas.rectangle(coords, outline="#00e5ff", width=line_width)
        canvas.text((coords[0] + 4, max(0, coords[1] - 18)), label, fill="#00e5ff")
    return annotated


st.set_page_config(page_title="X-ray Detector", page_icon="🩻", layout="wide")
st.title("🩻 X-ray Detector")
st.caption("Upload an X-ray image to identify objects using the trained YOLO model.")

with st.sidebar:
    st.header("Connection")
    api_url = st.text_input("API URL", value=DEFAULT_API_URL).rstrip("/")
    if st.button("Check API"):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            response.raise_for_status()
            st.success("API is ready")
        except requests.RequestException as exc:
            st.error(f"Cannot reach API: {exc}")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    raw_image = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(raw_image)).convert("RGB")
    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.image(image, use_container_width=True)

    if st.button("Analyze image", type="primary"):
        with st.spinner("Running detection..."):
            try:
                response = requests.post(
                    f"{api_url}/predict",
                    files={"file": (uploaded_file.name, raw_image, uploaded_file.type)},
                    timeout=90,
                )
                response.raise_for_status()
                result = response.json()
            except requests.RequestException as exc:
                st.error(f"Analysis failed: {exc}")
            else:
                detections = result["detections"]
                with right:
                    st.subheader("Detections")
                    st.image(draw_detections(image, detections), use_container_width=True)
                st.metric("Objects detected", result["detection_count"])
                if detections:
                    st.dataframe(
                        [
                            {
                                "Label": item["label"],
                                "Confidence": f'{item["confidence"]:.1%}',
                                "Box": item["box"],
                            }
                            for item in detections
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No detections found in this image.")
else:
    st.info("Upload a JPG, PNG, or WEBP image to get started.")
