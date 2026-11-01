from pathlib import Path
import io

import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="Fruit Detection System",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / " best.pt "

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The model file 'best.pt' was not found. Add best.pt to the same "
            "GitHub repository folder as app.py."
        )
    return YOLO(str(MODEL_PATH))

st.title(" Fruit Detection System")
st.write(
    "Upload an image and the trained YOLO model will detect the fruit objects "
    "it has learned to recognize."
)

with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=1.00,
        value=0.25,
        step=0.05,
        help="Higher values show only more confident detections.",
    )
    st.caption("The model is loaded once and reused between interactions.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the YOLO model: {exc}")
    st.info(
        "Check that best.pt is committed to your repository and that "
        "requirements.txt is installed successfully."
    )
    st.stop()

# Show the class labels supplied by the trained model, when available.
model_names = model.names
if isinstance(model_names, dict):
    labels = [str(v) for _, v in sorted(model_names.items())]
else:
    labels = [str(v) for v in model_names]

with st.expander("Model information"):
    st.write(f"**Model:** `{MODEL_PATH.name}`")
    if labels:
        st.write("**Recognized classes:** " + ", ".join(labels))
    else:
        st.write("Class labels are provided by the trained model.")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("The uploaded file could not be read as an image.")
        st.stop()

    with st.spinner("Detecting objects..."):
        results = model.predict(
            source=image,
            conf=confidence,
            verbose=False,
        )

    result = results[0]
    annotated = result.plot()
    # Ultralytics returns a BGR numpy array; convert to RGB for Streamlit.
    import cv2
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    left, right = st.columns(2)
    with left:
        st.subheader("Original image")
        st.image(image, use_container_width=True)
    with right:
        st.subheader("Detection result")
        st.image(annotated_rgb, use_container_width=True)

    boxes = result.boxes
    count = 0 if boxes is None else len(boxes)
    st.subheader("Detection summary")
    st.metric("Objects detected", count)

    if count == 0:
        st.warning(
            "No objects were detected. Try a clearer image or lower the "
            "confidence threshold."
        )
    else:
        rows = []
        for box in boxes:
            class_id = int(box.cls.item())
            score = float(box.conf.item())
            class_name = (
                model.names.get(class_id, str(class_id))
                if isinstance(model.names, dict)
                else model.names[class_id]
            )
            xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            rows.append(
                {
                    "Detected class": str(class_name),
                    "Confidence": f"{score:.1%}",
                    "Bounding box (x1, y1, x2, y2)": str(xyxy),
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

    # Provide the annotated image as a download.
    output = io.BytesIO()
    Image.fromarray(annotated_rgb).save(output, format="PNG")
    st.download_button(
        label="Download detected image",
        data=output.getvalue(),
        file_name="fruit_detection_result.png",
        mime="image/png",
    )
else:
    st.info("Upload a JPG, JPEG, PNG, or WEBP image to begin.")

st.divider()
st.caption("Powered by your trained Ultralytics YOLO model.")
