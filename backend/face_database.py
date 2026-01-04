import os
import json
import numpy as np
import base64
from datetime import datetime


class FaceDatabase:
    def __init__(self, db_path="face_database.json"):
        self.db_path = db_path
        self.faces = []
        self.load_database()

    def load_database(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.faces = data.get('faces', [])
                    for face in self.faces:
                        if 'features' in face:
                            face['features'] = np.array(face['features'])
            except Exception as e:
                print(f"Error loading database: {e}")
                self.faces = []

    def save_database(self):
        try:
            save_data = {'faces': []}
            for face in self.faces:
                face_data = face.copy()
                if isinstance(face_data.get('features'), np.ndarray):
                    face_data['features'] = face_data['features'].tolist()
                save_data['faces'].append(face_data)

            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Error saving database: {e}")
            return False

    def add_face(self, name, features, face_image=None, description=""):
        existing = self.find_face_by_name(name)
        if existing:
            existing['features'] = features
            existing['face_image'] = face_image
            existing['description'] = description
            existing['updated_at'] = datetime.now().isoformat()
            face_id = existing['id']
        else:
            face_id = len(self.faces) + 1
            face_record = {
                'id': face_id,
                'name': name,
                'features': features,
                'face_image': face_image,
                'description': description,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            self.faces.append(face_record)

        self.save_database()
        return face_id

    def find_face_by_name(self, name):
        for face in self.faces:
            if face['name'].lower() == name.lower():
                return face
        return None

    def find_face_by_id(self, face_id):
        for face in self.faces:
            if face['id'] == face_id:
                return face
        return None

    def recognize_face(self, features, threshold=0.4):
        if not self.faces:
            print("Database empty")
            return None, 0.0

        best_match = None
        best_similarity = 0.0
        best_distance = float('inf')

        for face in self.faces:
            if 'features' in face and face['features'] is not None:
                try:
                    distance = np.linalg.norm(features - face['features'])
                    print(f"Comparing: {face['name']}, Dist: {distance:.4f}")

                    if distance < best_distance:
                        best_distance = distance
                        best_similarity = max(0, 1 - (distance / 1.2))
                        best_match = face
                except Exception as e:
                    print(f"Comparison error: {e}")

        print(
            f"Match: {best_match['name'] if best_match else 'None'}, Dist: {best_distance:.4f}, Sim: {best_similarity:.4f}")

        if best_similarity >= threshold:
            return best_match, best_similarity
        else:
            return None, best_similarity

    def get_all_faces(self):
        return self.faces.copy()

    def delete_face(self, face_id):
        for i, face in enumerate(self.faces):
            if face['id'] == face_id:
                del self.faces[i]
                self.save_database()
                return True
        return False

    def clear_database(self):
        self.faces = []
        self.save_database()
        return True