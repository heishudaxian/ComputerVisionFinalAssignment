from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
import os
import cv2
import numpy as np
import base64
from face_detector import FaceDetector
from face_database import FaceDatabase

app = Flask(__name__, static_folder='../static', static_url_path='/static')
CORS(app)

if not os.path.exists('../static'):
    os.makedirs('../static')

face_detector = FaceDetector()
face_db = FaceDatabase()


@app.route('/api/detect', methods=['POST'])
def detect_faces():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        image_data = base64.b64decode(data['image'])
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Image decoding failed'}), 400

        faces = face_detector.detect_faces(image)

        result = []
        for face in faces:
            result.append({
                'box': face['box'],
                'confidence': face['confidence']
            })

        return jsonify({
            'success': True,
            'faces': result,
            'count': len(result)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/register', methods=['POST'])
def register_face():
    try:
        data = request.json
        if not data or 'image' not in data or 'name' not in data:
            return jsonify({'error': 'Missing required parameters'}), 400

        image_data = base64.b64decode(data['image'])
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Image decoding failed'}), 400

        faces = face_detector.detect_faces(image)

        if not faces:
            return jsonify({'error': 'No face detected'}), 400

        if len(faces) > 1:
            return jsonify({'error': 'Multiple faces detected. Please use a single face image.'}), 400

        face = faces[0]
        features = face_detector.extract_features(face['roi'])

        if features is None:
            return jsonify({'error': 'Feature extraction failed'}), 400

        face_roi = face['roi']
        _, buffer = cv2.imencode('.jpg', face_roi)
        face_image_base64 = base64.b64encode(buffer).decode('utf-8')

        name = data['name']
        description = data.get('description', '')
        face_id = face_db.add_face(name, features, face_image_base64, description)

        return jsonify({
            'success': True,
            'message': 'Registration successful',
            'face_id': face_id,
            'name': name
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        image_data = base64.b64decode(data['image'])
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Image decoding failed'}), 400

        faces = face_detector.detect_faces(image)

        if not faces:
            return jsonify({
                'success': True,
                'message': 'No face detected',
                'results': []
            })

        results = []
        for face in faces:
            features = face_detector.extract_features(face['roi'])

            if features is not None:
                matched_face, similarity = face_db.recognize_face(features)

                if matched_face:
                    results.append({
                        'box': face['box'],
                        'matched': True,
                        'name': matched_face['name'],
                        'face_id': matched_face['id'],
                        'similarity': float(similarity),
                        'confidence': face['confidence']
                    })
                else:
                    results.append({
                        'box': face['box'],
                        'matched': False,
                        'name': 'Unknown',
                        'similarity': 0.0,
                        'confidence': face['confidence']
                    })
            else:
                results.append({
                    'box': face['box'],
                    'matched': False,
                    'name': 'Unknown',
                    'similarity': 0.0,
                    'confidence': face['confidence']
                })

        return jsonify({
            'success': True,
            'results': results,
            'total_faces': len(results)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/faces', methods=['GET'])
def get_all_faces():
    try:
        faces = face_db.get_all_faces()

        result = []
        for face in faces:
            face_data = face.copy()
            if 'features' in face_data:
                del face_data['features']
            result.append(face_data)

        return jsonify({
            'success': True,
            'faces': result,
            'count': len(result)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/faces/<int:face_id>', methods=['DELETE'])
def delete_face(face_id):
    try:
        success = face_db.delete_face(face_id)

        if success:
            return jsonify({
                'success': True,
                'message': 'Face deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Face ID not found'
            }), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/test', methods=['GET'])
def test_api():
    return jsonify({
        'success': True,
        'message': 'Face recognition server is running',
        'version': '1.0.0'
    })


@app.route('/')
def serve_index():
    index_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'index.html')
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return render_template_string(html_content)
    else:
        return "index.html not found.", 404


if __name__ == '__main__':
    index_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'index.html')
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            content = f.read()
        updated_content = content.replace('const API_BASE_URL = "http://127.0.0.1:5000";', 'const API_BASE_URL = "";')
        if content != updated_content:
            with open(index_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            print("Frontend API_BASE_URL updated.")

    app.run(host='0.0.0.0', port=5000, debug=True)