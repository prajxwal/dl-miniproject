"""
Flask server for MNIST digit recognition.
Serves the frontend and handles prediction requests from the canvas via ONNX runtime.
"""

import base64
import io
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, ImageFilter
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)

# ─── Load Model ───
try:
    session = ort.InferenceSession('model/mnist_cnn.onnx')
    input_name = session.get_inputs()[0].name
    print('ONNX Model loaded successfully!')
except Exception as e:
    print(f'Model file not found or failed to load: {e}')
    session = None


def preprocess_image(image_data: str) -> np.ndarray:
    """
    Preprocess a base64-encoded canvas image to match MNIST format.
    Returns a numpy array suitable for ONNX inference.
    """
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    
    image_array = np.array(image)
    
    if image_array.mean() > 128:
        image = ImageOps.invert(image)
        image_array = np.array(image)
    
    rows = np.any(image_array > 30, axis=1)
    cols = np.any(image_array > 30, axis=0)
    
    if not rows.any() or not cols.any():
        return np.zeros((1, 1, 28, 28), dtype=np.float32)
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    padding = 30
    rmin = max(0, rmin - padding)
    rmax = min(image_array.shape[0], rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(image_array.shape[1], cmax + padding)
    
    cropped = image.crop((cmin, rmin, cmax, rmax))
    
    width, height = cropped.size
    max_dim = max(width, height)
    padded = Image.new('L', (max_dim, max_dim), 0)
    offset = ((max_dim - width) // 2, (max_dim - height) // 2)
    padded.paste(cropped, offset)
    
    # Use Resampling.LANCZOS to stay compatible with new PIL versions
    resized = padded.resize((28, 28), getattr(Image, 'Resampling', Image).LANCZOS)
    resized = resized.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    tensor = np.array(resized, dtype=np.float32) / 255.0
    tensor = (tensor - 0.1307) / 0.3081
    
    tensor = np.expand_dims(tensor, axis=0)
    tensor = np.expand_dims(tensor, axis=0)
    
    return tensor


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


# ─── Routes ───

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)


@app.route('/predict', methods=['POST'])
def predict():
    """Receive canvas image and return digit probabilities using ONNX."""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        if session is None:
            return jsonify({'error': 'ONNX model not loaded properly'}), 500
        
        tensor = preprocess_image(image_data)
        
        # Predict with ONNX Runtime
        outputs = session.run(None, {input_name: tensor})
        output = outputs[0]
        
        probabilities = softmax(output)[0]
        
        predicted_digit = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_digit])
        
        return jsonify({
            'digit': predicted_digit,
            'confidence': confidence,
            'probabilities': probabilities[:10].tolist()  # Only send 0-9 digit probs to the frontend
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print('\n🚀 Starting MNIST Digit Recognizer Server (ONNX)...')
    app.run(debug=True, port=5000)
