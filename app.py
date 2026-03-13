"""
Flask server for MNIST digit recognition.
Serves the frontend and handles prediction requests from the canvas.
"""

import base64
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)

# ─── Device ───
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── Model Definition (must match training notebook) ───
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


# ─── Load Model ───
model = MNISTNet().to(device)
try:
    model.load_state_dict(torch.load('model/mnist_cnn.pth', map_location=device))
    model.eval()
    print('✅ Model loaded successfully!')
except FileNotFoundError:
    print('⚠️  Model file not found at model/mnist_cnn.pth')
    print('   Run the training notebook first to generate the model.')


def preprocess_image(image_data: str) -> torch.Tensor:
    """
    Preprocess a base64-encoded canvas image to match MNIST format.
    
    Steps:
    1. Decode base64 → PIL Image
    2. Convert to grayscale
    3. Invert colors (canvas: white-on-black → MNIST: white-on-black)
    4. Find bounding box of the digit and crop with padding
    5. Resize to 28×28
    6. Normalize with MNIST stats
    """
    # Decode base64 image
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    
    # Invert: canvas has white drawing on black background
    # MNIST has white digits on black background — so this matches already
    # But the canvas alpha channel might cause issues, so let's handle it
    image_array = np.array(image)
    
    # If the image is mostly white (background), invert it
    if image_array.mean() > 128:
        image = ImageOps.invert(image)
        image_array = np.array(image)
    
    # Find bounding box of non-zero pixels (the digit)
    rows = np.any(image_array > 30, axis=1)
    cols = np.any(image_array > 30, axis=0)
    
    if not rows.any() or not cols.any():
        # Empty canvas — return zeros
        return torch.zeros(1, 1, 28, 28).to(device)
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Crop with some padding
    padding = 30
    rmin = max(0, rmin - padding)
    rmax = min(image_array.shape[0], rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(image_array.shape[1], cmax + padding)
    
    cropped = image.crop((cmin, rmin, cmax, rmax))
    
    # Make it square by adding padding
    width, height = cropped.size
    max_dim = max(width, height)
    padded = Image.new('L', (max_dim, max_dim), 0)
    offset = ((max_dim - width) // 2, (max_dim - height) // 2)
    padded.paste(cropped, offset)
    
    # Resize to 28×28 with antialiasing
    resized = padded.resize((28, 28), Image.LANCZOS)
    
    # Apply slight Gaussian blur (MNIST digits have soft edges)
    resized = resized.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Convert to tensor and normalize with MNIST stats
    tensor = torch.FloatTensor(np.array(resized)).unsqueeze(0).unsqueeze(0) / 255.0
    tensor = (tensor - 0.1307) / 0.3081
    
    return tensor.to(device)


# ─── Routes ───

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)


@app.route('/predict', methods=['POST'])
def predict():
    """Receive canvas image and return digit probabilities."""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Preprocess
        tensor = preprocess_image(image_data)
        
        # Predict
        with torch.no_grad():
            output = model(tensor)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
        
        # Get prediction
        predicted_digit = int(probabilities.argmax())
        confidence = float(probabilities[predicted_digit])
        
        return jsonify({
            'digit': predicted_digit,
            'confidence': confidence,
            'probabilities': probabilities.tolist()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print('\n🚀 Starting MNIST Digit Recognizer Server...')
    print('   Open http://localhost:5000 in your browser\n')
    app.run(debug=True, port=5000)
