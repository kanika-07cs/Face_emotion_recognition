import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model   # ✅ FIXED IMPORT
import os

# ---------------------------------------------------
# Load CNN model
# ---------------------------------------------------
MODEL_PATH = "face_reg_cnn_model.h5"

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found: {MODEL_PATH}")
    st.stop()

model = load_model(MODEL_PATH)

# ---------------------------------------------------
# Load Haar Cascade for face detection
# ---------------------------------------------------
# NOTE: Some installs (e.g., opencv-python-headless) may not ship Haar XMLs.
# Fallback to a local XML if provided.
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
LOCAL_CASCADE_PATH = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty() and os.path.exists(LOCAL_CASCADE_PATH):
    face_cascade = cv2.CascadeClassifier(LOCAL_CASCADE_PATH)



# ---------------------------------------------------
# Streamlit UI Setup
# ---------------------------------------------------
st.set_page_config(
    page_title="Face Emotion Recognition",
    page_icon="😊",
    layout="centered"
)

st.title("😊 Face Emotion Recognition App")
st.write(
    "Upload an image or take a photo to detect faces and predict emotions with confidence levels."
)

# Reset button
if st.button("🔄 Try Another Image"):
    st.session_state.clear()
    st.rerun()

# ---------------------------------------------------
# Option to upload OR take photo
# ---------------------------------------------------
option = st.radio("Choose Input Method:", ("📁 Upload Image", "📷 Open Camera"))

img = None

if option == "📁 Upload Image":
    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8
        )
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

elif option == "📷 Open Camera":
    captured_image = st.camera_input("Take a photo using your webcam")
    if captured_image:
        file_bytes = np.asarray(
            bytearray(captured_image.read()), dtype=np.uint8
        )
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# ---------------------------------------------------
# Face Detection & Emotion Prediction
# ---------------------------------------------------
if img is not None:
    # Robust guards to avoid OpenCV assertion: !empty() in detectMultiScale
    if face_cascade.empty():
        st.error("❌ Haar cascade XML failed to load.")
        st.stop()

    if img.size == 0:
        st.error("❌ Invalid/empty image received.")
        st.stop()

    # Some imdecode failures can result in None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray is None or gray.size == 0:
        st.error("❌ Failed to convert image to grayscale.")
        st.stop()


    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(40, 40)
    )


    if len(faces) == 0:
        st.warning("⚠️ No face detected. Please try again or use a clearer image.")
        st.image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            caption="Uploaded / Captured Image",
            use_container_width=True
        )

    else:
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        emotion_results = []

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = face / 255.0
            face = np.expand_dims(face, axis=0)

            pred = model.predict(face, verbose=0)
            emotion_idx = np.argmax(pred)
            confidence = float(np.max(pred))
            emotion = emotions[emotion_idx]

            cv2.rectangle(
                img, (x, y), (x + w, y + h),
                (0, 255, 0), 2
            )
            cv2.putText(
                img,
                f"{emotion} ({confidence*100:.1f}%)",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            emotion_results.append((emotion, confidence))

        st.image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            caption="Detected Faces & Emotions",
            use_container_width=True
        )

        st.subheader("📊 Emotion Confidence Scores")
        for i, (emotion, conf) in enumerate(emotion_results, 1):
            st.write(f"**Face {i}: {emotion} ({conf*100:.1f}%)**")
            st.progress(conf)
