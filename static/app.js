/**
 * MNIST Digit Recognizer — Frontend Logic
 * 
 * Handles canvas drawing (mouse + touch), sends drawings to the
 * Flask backend for prediction, and animates probability bars.
 */

// ─── DOM Elements ───
const canvas = document.getElementById('draw-canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clear-btn');
const undoBtn = document.getElementById('undo-btn');
const brushSizeInput = document.getElementById('brush-size');
const brushSizeValue = document.getElementById('brush-size-value');
const canvasHint = document.getElementById('canvas-hint');
const predictionDigit = document.getElementById('prediction-digit');
const predictionConfidence = document.getElementById('prediction-confidence');
const probabilityBarsContainer = document.getElementById('probability-bars');
const predictionCard = document.querySelector('.prediction-card');
const canvasWrapper = document.querySelector('.canvas-wrapper');

// ─── State ───
let isDrawing = false;
let lastX = 0;
let lastY = 0;
let brushSize = 18;
let hasDrawn = false;
let predictTimeout = null;
let strokes = [];       // Array of ImageData snapshots for undo
let currentStroke = [];  // Points in current stroke

// ─── Initialize ───
function init() {
    setupCanvas();
    createProbabilityBars();
    setupEventListeners();
}

function setupCanvas() {
    // Set canvas resolution to match display size
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    
    // Canvas settings
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, rect.width, rect.height);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = brushSize;
}

function createProbabilityBars() {
    probabilityBarsContainer.innerHTML = '';
    for (let i = 0; i < 10; i++) {
        const row = document.createElement('div');
        row.className = 'prob-row';
        row.innerHTML = `
            <span class="prob-digit" id="prob-digit-${i}">${i}</span>
            <div class="prob-bar-track">
                <div class="prob-bar-fill bar-${i}" id="prob-bar-${i}"></div>
            </div>
            <span class="prob-value" id="prob-value-${i}">0.0%</span>
        `;
        probabilityBarsContainer.appendChild(row);
    }
}

// ─── Event Listeners ───
function setupEventListeners() {
    // Mouse events
    canvas.addEventListener('mousedown', startDraw);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', endDraw);
    canvas.addEventListener('mouseleave', endDraw);

    // Touch events
    canvas.addEventListener('touchstart', handleTouchStart, { passive: false });
    canvas.addEventListener('touchmove', handleTouchMove, { passive: false });
    canvas.addEventListener('touchend', endDraw);
    canvas.addEventListener('touchcancel', endDraw);

    // Buttons
    clearBtn.addEventListener('click', clearCanvas);
    undoBtn.addEventListener('click', undoStroke);

    // Brush size
    brushSizeInput.addEventListener('input', (e) => {
        brushSize = parseInt(e.target.value);
        brushSizeValue.textContent = `${brushSize}px`;
        ctx.lineWidth = brushSize;
    });

    // Resize handler
    window.addEventListener('resize', debounce(() => {
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        setupCanvas();
        ctx.putImageData(imageData, 0, 0);
    }, 250));
}

// ─── Drawing Functions ───
function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
    };
}

function startDraw(e) {
    e.preventDefault();
    isDrawing = true;
    
    // Save state before this stroke for undo
    const dpr = window.devicePixelRatio || 1;
    strokes.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
    
    // Limit undo history to 20 strokes
    if (strokes.length > 20) strokes.shift();
    
    const pos = getPos(e);
    lastX = pos.x;
    lastY = pos.y;

    // Draw a dot for single clicks
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, brushSize / 2, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();

    // Hide hint
    if (!hasDrawn) {
        hasDrawn = true;
        canvasHint.classList.add('hidden');
    }

    canvasWrapper.classList.add('drawing');

    // Schedule prediction
    schedulePrediction();
}

function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();

    const pos = getPos(e);

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();

    lastX = pos.x;
    lastY = pos.y;

    // Schedule prediction with debounce
    schedulePrediction();
}

function endDraw(e) {
    if (!isDrawing) return;
    isDrawing = false;
    canvasWrapper.classList.remove('drawing');

    // Final prediction after stroke ends
    schedulePrediction(100);
}

// ─── Touch Handlers ───
function handleTouchStart(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent('mousedown', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    startDraw(mouseEvent);
}

function handleTouchMove(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent('mousemove', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    draw(mouseEvent);
}

// ─── Canvas Actions ───
function clearCanvas() {
    const rect = canvas.getBoundingClientRect();
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, rect.width, rect.height);
    strokes = [];
    hasDrawn = false;
    canvasHint.classList.remove('hidden');
    
    // Reset predictions
    predictionDigit.textContent = '?';
    predictionConfidence.textContent = 'Draw a digit to start';
    predictionCard.classList.remove('active');
    
    for (let i = 0; i < 10; i++) {
        document.getElementById(`prob-bar-${i}`).style.width = '0%';
        document.getElementById(`prob-value-${i}`).textContent = '0.0%';
        document.getElementById(`prob-value-${i}`).classList.remove('highlight');
        document.getElementById(`prob-digit-${i}`).classList.remove('highlight');
    }
}

function undoStroke() {
    if (strokes.length === 0) return;
    
    const previous = strokes.pop();
    ctx.putImageData(previous, 0, 0);
    
    // Check if canvas is now empty
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let hasContent = false;
    for (let i = 0; i < imageData.data.length; i += 4) {
        if (imageData.data[i] > 10 || imageData.data[i + 1] > 10 || imageData.data[i + 2] > 10) {
            hasContent = true;
            break;
        }
    }
    
    if (hasContent) {
        schedulePrediction(100);
    } else {
        hasDrawn = false;
        canvasHint.classList.remove('hidden');
        predictionDigit.textContent = '?';
        predictionConfidence.textContent = 'Draw a digit to start';
        predictionCard.classList.remove('active');
        for (let i = 0; i < 10; i++) {
            document.getElementById(`prob-bar-${i}`).style.width = '0%';
            document.getElementById(`prob-value-${i}`).textContent = '0.0%';
            document.getElementById(`prob-value-${i}`).classList.remove('highlight');
            document.getElementById(`prob-digit-${i}`).classList.remove('highlight');
        }
    }
}

// ─── Prediction ───
function schedulePrediction(delay = 200) {
    if (predictTimeout) clearTimeout(predictTimeout);
    predictTimeout = setTimeout(predict, delay);
}

async function predict() {
    if (!hasDrawn) return;

    // Get canvas data as base64
    // First, create a temporary canvas at display resolution
    const rect = canvas.getBoundingClientRect();
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = rect.width;
    tempCanvas.height = rect.height;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(canvas, 0, 0, rect.width, rect.height);
    
    const imageData = tempCanvas.toDataURL('image/png');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });

        if (!response.ok) throw new Error('Prediction failed');

        const result = await response.json();
        updatePrediction(result);
    } catch (err) {
        console.error('Prediction error:', err);
    }
}

function updatePrediction(result) {
    const { digit, confidence, probabilities } = result;

    // Update main prediction
    const prevDigit = predictionDigit.textContent;
    predictionDigit.textContent = digit;
    predictionCard.classList.add('active');
    
    // Bounce animation on digit change
    if (prevDigit !== String(digit)) {
        predictionDigit.classList.remove('bounce');
        void predictionDigit.offsetWidth; // trigger reflow
        predictionDigit.classList.add('bounce');
    }

    predictionConfidence.innerHTML = `Confidence: <span class="confidence-value">${(confidence * 100).toFixed(1)}%</span>`;

    // Update probability bars
    const maxProb = Math.max(...probabilities);
    
    for (let i = 0; i < 10; i++) {
        const bar = document.getElementById(`prob-bar-${i}`);
        const value = document.getElementById(`prob-value-${i}`);
        const digitLabel = document.getElementById(`prob-digit-${i}`);
        const prob = probabilities[i];
        
        // Animate bar width (scale to make even small probabilities visible)
        bar.style.width = `${prob * 100}%`;
        
        // Update text
        value.textContent = `${(prob * 100).toFixed(1)}%`;
        
        // Highlight the predicted digit
        if (i === digit) {
            value.classList.add('highlight');
            digitLabel.classList.add('highlight');
        } else {
            value.classList.remove('highlight');
            digitLabel.classList.remove('highlight');
        }
    }
}

// ─── Utilities ───
function debounce(fn, delay) {
    let timeout;
    return (...args) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => fn(...args), delay);
    };
}

// ─── Start ───
document.addEventListener('DOMContentLoaded', init);
