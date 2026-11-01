
import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

# ---------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Face Emotion Recognition",
    page_icon="😊",
    layout="centered"
)

# ---------------------------------------------------
# Project Paths
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "face_reg_cnn_model.h5"
)

LOCAL_CASCADE_PATH = os.path.join(
    BASE_DIR,
    "haarcascade_frontalface_default.xml"
)

# ---------------------------------------------------
# Load CNN Model
# ---------------------------------------------------
@st.cache_resource
def load_emotion_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    return load_model(MODEL_PATH)

model = load_emotion_model()

# ---------------------------------------------------
# Load Haar Cascade
# ---------------------------------------------------
@st.cache_resource
def load_face_cascade():

    # First try the local XML file
    if os.path.exists(LOCAL_CASCADE_PATH):
        cascade = cv2.CascadeClassifier(LOCAL_CASCADE_PATH)

        if not cascade.empty():
            return cascade

    # Fallback to OpenCV bundled Haar Cascade
    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    cascade = cv2.CascadeClassifier(cascade_path)

    if cascade.empty():
        raise RuntimeError(
            "Unable to load Haar Cascade XML file."
        )

    return cascade


try:
    face_cascade = load_face_cascade()
except Exception as e:
    st.error(f"Face detection initialization failed: {e}")
    st.stop()

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.title("😊 Face Emotion Recognition App")

st.write(
    "Upload an image or take a photo to detect faces "
    "and predict emotions with confidence levels."
)

# Reset Button
if st.button("🔄 Try Another Image"):
    st.session_state.clear()
    st.rerun()

# ---------------------------------------------------
# Input Selection
# ---------------------------------------------------
option = st.radio(
    "Choose Input Method:",
    ("📁 Upload Image", "📷 Open Camera")
)

img = None

if option == "📁 Upload Image":

    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        img = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

elif option == "📷 Open Camera":

    captured_image = st.camera_input(
        "Take a photo using your webcam"
    )

    if captured_image is not None:

        file_bytes = np.asarray(
            bytearray(captured_image.read()),
            dtype=np.uint8
        )

        img = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

# ---------------------------------------------------
# Face Detection & Emotion Prediction
# ---------------------------------------------------
if img is not None:

    if img.size == 0:
        st.error("Invalid or empty image received.")
        st.stop()

    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:

        st.warning(
            "⚠️ No face detected. "
            "Please try another clear image."
        )

        st.image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            caption="Uploaded / Captured Image",
            use_container_width=True
        )

    else:

        emotions = [
            'Angry',
            'Disgust',
            'Fear',
            'Happy',
            'Neutral',
            'Sad',
            'Surprise'
        ]

        emotion_results = []

        for (x, y, w, h) in faces:

            # Extract detected face
            face = img[y:y+h, x:x+w]

            # Resize to CNN input dimensions
            face = cv2.resize(
                face,
                (48, 48)
            )

            # Convert BGR to RGB
            face = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            )

            # Normalize pixel values
            face = face.astype("float32") / 255.0

            # Add batch dimension
            face = np.expand_dims(
                face,
                axis=0
            )

            # Predict emotion
            pred = model.predict(
                face,
                verbose=0
            )

            emotion_idx = int(np.argmax(pred))
            confidence = float(np.max(pred))

            emotion = emotions[emotion_idx]

            # Draw rectangle around face
            cv2.rectangle(
                img,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Display predicted emotion
            cv2.putText(
                img,
                f"{emotion} ({confidence * 100:.1f}%)",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            emotion_results.append(
                (emotion, confidence)
            )

        # Display processed image
        st.image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            caption="Detected Faces & Emotions",
            use_container_width=True
        )

        # Display confidence scores
        st.subheader("📊 Emotion Confidence Scores")

        for i, (emotion, conf) in enumerate(
            emotion_results, 1
        ):

            st.write(
                f"**Face {i}: {emotion} "
                f"({conf * 100:.1f}%)**"
            )

            st.progress(conf)

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("---")
st.caption(
    "Face Emotion Recognition using CNN and OpenCV"
)
