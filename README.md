# 🖐 AI Hand Gesture Recognition System

A Deep Learning based Hand Gesture Recognition Web Application developed using TensorFlow, CNN, OpenCV, and Flask.

This project can recognize different hand gestures from uploaded images and display the predicted gesture along with an emoji representation.

---

## 📌 Project Overview

Hand Gesture Recognition is an important Computer Vision application that enables Human-Computer Interaction (HCI).

This system uses a Convolutional Neural Network (CNN) trained on the LeapGestRecog dataset to classify hand gestures into 10 different categories.

The trained model is integrated with a Flask web application where users can upload gesture images and receive instant predictions.

---

## 🎯 Features

- Upload hand gesture images
- Real-time gesture prediction
- CNN-based deep learning model
- Flask web application
- Image preview
- Emoji-based gesture display
- Modern responsive UI
- Trained TensorFlow model

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-Learn
- Flask
- HTML
- CSS

---

## 📂 Dataset

Dataset Used:

**LeapGestRecog Dataset**

https://www.kaggle.com/datasets/gti-upm/leapgestrecog

Dataset contains 10 gesture classes:

| Label | Gesture |
|---------|---------|
| 01 | Palm |
| 02 | L |
| 03 | Fist |
| 04 | Fist Moved |
| 05 | Thumb |
| 06 | Index |
| 07 | OK |
| 08 | Palm Moved |
| 09 | C |
| 10 | Down |

---

## 🧠 Model Architecture

CNN Architecture:

Input Layer (64x64x3)

↓
Conv2D (32 Filters)

↓
MaxPooling2D

↓
Conv2D (64 Filters)

↓
MaxPooling2D

↓
Flatten

↓
Dense (128)

↓
Output Layer (10 Classes)

---

## 📁 Project Structure

```text
PRODIGY_ML_04/
│
├── data/
│   └── leapGestRecog/
│
├── model/
│   └── hand_gesture_model.h5
│
├── static/
│   ├── style.css
│   └── uploads/
│
├── templates/
│   └── index.html
│
├── flask-app.py
├── train_model.py
├── requirements.txt
├── README.md
└── screenshots/
```

---

## ⚙️ Installation

### 1 Clone Repository

```bash
git clone https://github.com/yourusername/PRODIGY_ML_04.git
```

### 2 Navigate to Project Folder

```bash
cd PRODIGY_ML_04
```

### 3 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Train Model

Run:

```bash
python train_model.py
```

After training, model will be saved as:

```text
model/hand_gesture_model.h5
```

---

## 🌐 Run Flask Application

```bash
python flask-app.py
```

Open browser:

```text
http://127.0.0.1:5000
```

---

## 📊 Results

- Training Accuracy: 99.9%
- Validation Accuracy: 100%
- Gesture Classes: 10

---

## 📷 Screenshots

### Home Page

(Add Screenshot Here)

### Prediction Result

(Add Screenshot Here)

---

## 🔮 Future Improvements

- Real-time Webcam Recognition
- Live Gesture Tracking
- Mobile Responsive Design
- Gesture History Tracking
- Voice Feedback System

---

## 👨‍💻 Author

Developed as part of the **Prodigy InfoTech Machine Learning Internship**.

### Task-04
Hand Gesture Recognition using CNN, TensorFlow, OpenCV and Flask.

---

## 📜 License

This project is for educational and internship purposes.
