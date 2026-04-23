"""
Flask server for MNIST digit recognition.
Serves the frontend and handles prediction requests from the canvas via PyTorch.
"""

import base64
import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageFilter
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)

# ─── Model Architecture (must match training notebook exactly) ───
class MNISTNet(nn.Module):
    """Deeper VGG-style CNN for MNIST digit classification + junk rejection."""

    def __init__(self, num_classes=11):
        super(MNISTNet, self).__init__()

        # Block 1: 1 -> 32
        self.conv1a = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(32)

        # Block 2: 32 -> 64
        self.conv2a = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)

        # Block 3: 64 -> 128
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(0.25)
        self.dropout_fc = nn.Dropout(0.5)

        # After 3 pools: 28->14->7->3, so 128*3*3 = 1152
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)

        # Block 2
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)

        # Block 3
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)

        # Flatten + FC
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x


# ─── Load Model ───
device = torch.device('cpu')  # CPU is fine for single-image inference
model = None

try:
    model = MNISTNet(num_classes=11).to(device)
    state_dict = torch.load('model/mnist_cnn.pth', map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print('[OK] PyTorch model loaded successfully!')
except Exception as e:
    print(f'[ERROR] Model file not found or failed to load: {e}')
    model = None


def preprocess_image(image_data: str) -> np.ndarray:
    """
    Preprocess a base64-encoded canvas image to match MNIST format.
    Returns a PyTorch tensor suitable for inference.
    """
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes)).convert('L')

    image_array = np.array(image)

    # Canvas is white-on-black by default, MNIST is also white-on-black
    # But if background is light, invert so digit strokes become white
    if image_array.mean() > 128:
        image = ImageOps.invert(image)
        image_array = np.array(image)

    # Find bounding box of the drawn content
    rows = np.any(image_array > 30, axis=1)
    cols = np.any(image_array > 30, axis=0)

    if not rows.any() or not cols.any():
        return torch.zeros(1, 1, 28, 28, dtype=torch.float32)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add padding around the digit
    padding = 30
    rmin = max(0, rmin - padding)
    rmax = min(image_array.shape[0], rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(image_array.shape[1], cmax + padding)

    cropped = image.crop((cmin, rmin, cmax, rmax))

    # Make it square by padding the shorter side
    width, height = cropped.size
    max_dim = max(width, height)
    padded = Image.new('L', (max_dim, max_dim), 0)
    offset = ((max_dim - width) // 2, (max_dim - height) // 2)
    padded.paste(cropped, offset)

    # Resize to 28x28 (MNIST dimensions)
    resized = padded.resize((28, 28), getattr(Image, 'Resampling', Image).LANCZOS)
    resized = resized.filter(ImageFilter.GaussianBlur(radius=0.5))

    # Normalize with MNIST statistics
    tensor = np.array(resized, dtype=np.float32) / 255.0
    tensor = (tensor - 0.1307) / 0.3081

    # Shape: [1, 1, 28, 28]  (batch, channel, height, width)
    tensor = torch.from_numpy(tensor).unsqueeze(0).unsqueeze(0)

    return tensor


# ─── Routes ───

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)


@app.route('/predict', methods=['POST'])
def predict():
    """Receive canvas image and return digit probabilities using PyTorch."""
    try:
        data = request.get_json()
        image_data = data.get('image')

        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400

        if model is None:
            return jsonify({'error': 'PyTorch model not loaded properly'}), 500

        tensor = preprocess_image(image_data).to(device)

        # Predict with PyTorch (no gradient needed for inference)
        with torch.no_grad():
            output = model(tensor)
            probabilities = F.softmax(output, dim=1)[0]

        probabilities_np = probabilities.cpu().numpy()
        predicted_digit = int(np.argmax(probabilities_np))
        confidence = float(probabilities_np[predicted_digit])

        return jsonify({
            'digit': predicted_digit,
            'confidence': confidence,
            'probabilities': probabilities_np[:10].tolist()  # Only send 0-9 digit probs to the frontend
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print('\n[*] Starting MNIST Digit Recognizer Server (PyTorch)...')
    app.run(debug=True, port=5000)
