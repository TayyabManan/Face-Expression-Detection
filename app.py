"""
Flask Web Application for Face Expression Detection
Showcases the trained ResNet model with 80% accuracy on RAF-DB dataset.
"""

import os
import io
import base64
import torch
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from torchvision import transforms
import cv2

# Try to import MTCNN (primary face detector)
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("Warning: facenet-pytorch not installed. Using Haar Cascade only.")
    print("Install with: pip install facenet-pytorch")

# Import model and config
from src.models.emotion_resnet import EmotionResNet
from config import EMOTION_LABELS, IMAGE_SIZE, DEVICE, MODELS_DIR

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Global variables
model = None
mtcnn = None

# Emotion colors for visualization (RGB)
EMOTION_COLORS = {
    "Surprise": (0, 255, 255),    # Cyan
    "Fear": (180, 0, 180),        # Purple
    "Disgust": (0, 180, 0),       # Green
    "Happiness": (255, 220, 0),   # Yellow
    "Sadness": (0, 100, 255),     # Blue
    "Anger": (255, 0, 0),         # Red
    "Neutral": (128, 128, 128)    # Gray
}

# Preprocessing transform (same as test transforms)
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model():
    """Load the trained emotion detection model and initialize MTCNN."""
    global model, mtcnn

    model_path = MODELS_DIR / "best_resnet_rafdb.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Initialize emotion model
    model = EmotionResNet(num_classes=7, dropout_rate=0.5, pretrained=False)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(DEVICE)
    model.eval()

    print(f"[Flask] Model loaded from {model_path}")
    print(f"[Flask] Using device: {DEVICE}")

    # Initialize MTCNN face detector (same settings as test_image.py)
    if MTCNN_AVAILABLE:
        mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,
            device=DEVICE,
            keep_all=True
        )
        print("[Flask] MTCNN face detector initialized")
    else:
        print("[Flask] Using Haar Cascade face detector (MTCNN not available)")

    return model


def detect_faces_mtcnn(image):
    """
    Detect faces using MTCNN (more accurate).
    Returns list of (x, y, w, h) tuples.
    """
    global mtcnn

    # Convert PIL to numpy RGB
    img_array = np.array(image)

    # Detect faces
    boxes, probs, landmarks = mtcnn.detect(img_array, landmarks=True)

    faces = []
    if boxes is not None:
        for i, box in enumerate(boxes):
            if probs[i] > 0.9:  # Only keep high confidence detections (same as test_image.py)
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                faces.append((int(x1), int(y1), int(w), int(h)))

    return faces


def detect_faces_haar(image):
    """
    Detect faces using OpenCV Haar Cascade (fallback).
    Returns list of (x, y, w, h) tuples.
    """
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Histogram equalization for better detection
    gray = cv2.equalizeHist(gray)

    # Load Haar Cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def detect_faces(image):
    """
    Detect faces using MTCNN (primary) or Haar Cascade (fallback).
    Returns list of (x, y, w, h) tuples.
    """
    if MTCNN_AVAILABLE and mtcnn is not None:
        faces = detect_faces_mtcnn(image)
        # Fall back to Haar if MTCNN finds nothing
        if len(faces) == 0:
            faces = detect_faces_haar(image)
        return faces
    else:
        return detect_faces_haar(image)


def predict_emotion(face_image):
    """
    Predict emotion for a face image.
    Returns (emotion_label, confidence, all_probabilities).
    """
    global model

    # Ensure RGB
    if face_image.mode != 'RGB':
        face_image = face_image.convert('RGB')

    # Preprocess
    input_tensor = preprocess(face_image).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # Get results
    predicted_class = predicted.item()
    confidence_score = confidence.item()
    all_probs = probabilities[0].cpu().numpy()

    emotion_label = EMOTION_LABELS[predicted_class]

    # Create probability dict
    prob_dict = {EMOTION_LABELS[i]: float(all_probs[i]) for i in range(len(EMOTION_LABELS))}

    return emotion_label, confidence_score, prob_dict


def process_image(image):
    """
    Process an image: detect faces and predict emotions.
    Returns annotated image and results.
    """
    # Detect faces
    faces = detect_faces(image)

    results = []
    img_array = np.array(image)

    # Process each face
    for i, (x, y, w, h) in enumerate(faces):
        # Add margin
        margin = int(0.1 * min(w, h))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img_array.shape[1], x + w + margin)
        y2 = min(img_array.shape[0], y + h + margin)

        # Extract face
        face_region = img_array[y1:y2, x1:x2]
        face_image = Image.fromarray(face_region)

        # Predict emotion
        emotion, confidence, probabilities = predict_emotion(face_image)

        results.append({
            'face_id': i + 1,
            'bbox': [int(x), int(y), int(w), int(h)],
            'emotion': emotion,
            'confidence': round(confidence * 100, 2),
            'probabilities': {k: round(v * 100, 2) for k, v in probabilities.items()}
        })

        # Draw on image
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))
        cv2.rectangle(img_array, (x, y), (x + w, y + h), color, 2)

        # Add label
        label = f"{emotion} ({confidence * 100:.1f}%)"
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Background for text
        cv2.rectangle(
            img_array,
            (x, y - text_height - 10),
            (x + text_width + 10, y),
            color,
            -1
        )

        # Text
        cv2.putText(
            img_array,
            label,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness
        )

    # Convert back to PIL
    annotated_image = Image.fromarray(img_array)

    return annotated_image, results


def image_to_base64(image):
    """Convert PIL image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        # Load image
        image = Image.open(file.stream)

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Process image
        annotated_image, results = process_image(image)

        # Convert to base64
        original_b64 = image_to_base64(image)
        annotated_b64 = image_to_base64(annotated_image)

        return jsonify({
            'success': True,
            'original_image': original_b64,
            'annotated_image': annotated_b64,
            'faces_detected': len(results),
            'results': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_single', methods=['POST'])
def predict_single():
    """Predict emotion for a single face image (no face detection)."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        # Load image
        image = Image.open(file.stream)

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Predict directly (assume image is already a face)
        emotion, confidence, probabilities = predict_emotion(image)

        return jsonify({
            'success': True,
            'emotion': emotion,
            'confidence': round(confidence * 100, 2),
            'probabilities': {k: round(v * 100, 2) for k, v in probabilities.items()}
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(DEVICE),
        'face_detector': 'MTCNN' if (MTCNN_AVAILABLE and mtcnn is not None) else 'Haar Cascade'
    })


# Load model on startup (works with both direct run and gunicorn)
load_model()

if __name__ == '__main__':
    # Determine detector name
    detector_name = "MTCNN" if (MTCNN_AVAILABLE and mtcnn is not None) else "Haar Cascade"

    # Run Flask app
    print("\n" + "="*50)
    print("Face Expression Detection Web App")
    print("="*50)
    print(f"Model: ResNet-18 (80% accuracy)")
    print(f"Device: {DEVICE}")
    print(f"Face Detector: {detector_name}")
    print(f"Emotions: {list(EMOTION_LABELS.values())}")
    print("="*50)
    print("Open http://localhost:5000 in your browser")
    print("="*50 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
