
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

st.set_page_config(
    page_title="Fruit Detection System",
    page_icon="🍎",
    layout="wide"
)

st.title("🍎 Fruit Detection System")
st.write("Upload an image to detect fruits using YOLO.")

@st.cache_resource
def load_model():
    return YOLO("yolo11m.pt")

try:
    model = load_model()
    st.success("YOLO model loaded successfully!")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

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
            results = model.predict(image, conf=0.25)

            result_image = results[0].plot()

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Original Image")

            with col2:
                st.image(
                    result_image,
                    caption="Detection Results"
                )

            st.write("### Detection Summary")

            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                st.write(
                    f"**{model.names[class_id]}** "
                    f"- Confidence: {confidence:.2%}"
                )
