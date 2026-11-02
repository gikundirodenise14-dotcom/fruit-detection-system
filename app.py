import io
from collections import Counter

import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="Fruit Detection System",
    page_icon="🍎",
    layout="wide"
)

st.title("🍎 Fruit Detection System")
st.write("Upload an image to detect **apples, bananas and oranges** using a trained YOLOv8 model.")


@st.cache_resource
def load_model():
    return YOLO("best.pt")   # your trained fruit model (was yolo11m.pt)


try:
    model = load_model()
    st.success("YOLO model loaded successfully!")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.40, 0.05)

uploaded_file = st.file_uploader(
    "Upload a fruit image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(
        io.BytesIO(uploaded_file.getvalue())
    ).convert("RGB")

    st.image(image, caption="Uploaded Image", width=500)

    if st.button("🔍 Detect Fruits"):
        with st.spinner("Detecting fruits..."):
            results = model.predict(image, conf=conf)

            # plot() returns BGR; flip to RGB so colours look correct
            result_image = results[0].plot()[:, :, ::-1]

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image")
        with col2:
            st.image(result_image, caption="Detection Results")

        st.write("### Detection Summary")

        boxes = results[0].boxes
        if len(boxes) == 0:
            st.warning("No fruit detected. Try lowering the confidence threshold.")
        else:
            counts = Counter(model.names[int(b.cls[0])] for b in boxes)
            cols = st.columns(len(counts))
            for c, (name, n) in zip(cols, counts.items()):
                c.metric(name.capitalize(), n)

            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                st.write(
                    f"**{model.names[class_id]}** "
                    f"- Confidence: {confidence:.2%}"
                )
