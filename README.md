
# 😊 Face Emotion Recognition

## 📌 Project Description

This project implements a **Face Emotion Recognition System** using machine learning and computer vision techniques.
The model identifies human emotions (such as Happy, Sad, Angry, Neutral, etc.) from facial images captured through a camera or uploaded by the user.

The project includes:

* Data preprocessing and analysis
* Model building and training
* Real-time emotion detection
* A **Streamlit web application** for interactive emotion prediction

This project is developed as a **mini project** to gain hands-on experience in deep learning and computer vision.

---

## 📁 Dataset Information

* **Dataset Used:** Facial Emotion Dataset (FER / custom dataset)

Typical emotion labels include:

* Happy
* Neutral
* Sad
* Angry
* Surprise
* Fear

---

## 🛠️ Technologies & Libraries Used

* Python
* OpenCV
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Streamlit

---

## 📂 Project Structure

```
Face-Emotion-Recognition
│
├── dataset/
├── face_emotion.ipynb
├── saved_model.h5
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Selvaganapathy-k/Face-Emotion-Recognition
cd Face-Emotion-Recognition
```

---

### 2️⃣ (Optional) Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Required Libraries

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the Streamlit Application Locally

```bash
streamlit run app.py
```

---

## 🌐 Live Application

🔗 **Streamlit App URL:**
👉 [https://selvaganapathy-k-face-emotion-recognition-app-vw73hv.streamlit.app/](https://selvaganapathy-k-face-emotion-recognition-app-vw73hv.streamlit.app/)

---

## 📸 Features

* Real-time emotion detection using webcam
* Supports image upload
* Predicts multiple facial emotions
* User-friendly Streamlit interface

---

## 🔍 Model Details

* Model Type: **Convolutional Neural Network (CNN)**
* Framework: **TensorFlow / Keras**
* Input: Facial images
* Output: Emotion class label

---

## 🎓 Learning Outcomes

* Understanding computer vision fundamentals
* Building CNN models for image classification
* Real-time prediction using webcam
* Deploying ML applications using Streamlit
* Structuring deep learning projects on GitHub

---

## 📌 Notes

* Virtual environment folders (`venv`, `myvenv`) are not included.
* All required dependencies are listed in `requirements.txt`.
* This project is for **educational purposes only**.

---

## 🏁 Conclusion

This project demonstrates an end-to-end **Face Emotion Recognition system**, combining deep learning, computer vision, and web deployment to deliver real-time emotion prediction.
