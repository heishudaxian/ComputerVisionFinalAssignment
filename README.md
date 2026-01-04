# Face Recognition System based on YOLO

This project implements a real-time face recognition system using YOLO for face detection and a Flask backend.

## 1. Requirements

- **Python**: 3.9
- **Core Libraries**:
  - `torch==2.4.0`
  - `torchvision==0.19.0`
  - `ultralytics`
  - `dlib==19.22.99`
  - `face-recognition==1.3.0`
  - `flask==3.0.0`
  - `flask-cors==4.0.0`
  - `opencv-python`
  - `numpy==1.24.3`

## 2. Pretrained Models

- **Detection Model**: The trained YOLO weights are located at `exp/weights/best.pt`.
- **Feature Extraction**: Uses the official `face-recognition-models`.

## 3. Preparation for Testing

### Step 1: Install Dependencies
install the necessary environment

### Step 2: Start the Backend Server

Navigate to the backend directory and run the application:

```bash
cd backend
python app.py
```

### Step 3: Run the Frontend
Open `index.html` in web browser.

Click "Start Camera" to begin real-time face detection and recognition.