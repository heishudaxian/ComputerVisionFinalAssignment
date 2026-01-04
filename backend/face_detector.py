import cv2
import numpy as np
from ultralytics import YOLO
import face_recognition
import os


class FaceDetector:
    def __init__(self, model_path="../exp/weights/best.pt"):
        self.model = YOLO(model_path)
        self.conf_threshold = 0.5

    def detect_faces(self, image):
        if len(image.shape) == 3 and image.shape[2] == 3:
            results = self.model(image)

            faces = []
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    if box.conf[0] > self.conf_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])

                        h, w = image.shape[:2]
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(w, x2)
                        y2 = min(h, y2)

                        face_roi = image[y1:y2, x1:x2]

                        faces.append({
                            'box': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'roi': face_roi
                        })

            return faces
        else:
            raise ValueError("Invalid image format. Provide BGR image.")

    def extract_features(self, face_roi):
        try:
            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_face)

            if face_encodings:
                return face_encodings[0]
            else:
                return None
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def compare_features(self, features1, features2):
        if features1 is None or features2 is None:
            print("Empty feature vectors")
            return 0.0

        try:
            distance = np.linalg.norm(features1 - features2)
            similarity = max(0, 1 - (distance / 1.2))

            print(f"Comparison - Dist: {distance:.4f}, Sim: {similarity:.4f}")
            return similarity
        except Exception as e:
            print(f"Comparison error: {e}")
            return 0.0

    def draw_detections(self, image, faces, names=None, similarities=None):
        if names is None:
            names = ['Unknown'] * len(faces)
        if similarities is None:
            similarities = [0.0] * len(faces)

        result_image = image.copy()
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face['box']
            confidence = face['confidence']
            name = names[i]
            similarity = similarities[i]

            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if similarity > 0:
                label = f"{name}: {similarity:.2f}"
            else:
                label = f"Face: {confidence:.2f}"

            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y1_label = max(y1 - 10, label_size[1])
            cv2.rectangle(result_image, (x1, y1_label - label_size[1]),
                          (x1 + label_size[0], y1_label + base_line), (0, 255, 0), cv2.FILLED)
            cv2.putText(result_image, label, (x1, y1_label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return result_image