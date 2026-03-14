# MNIST Digit Recognizer

A neural network that recognizes handwritten digits. A kindergartener can do this for free. I spent three weeks on it anyway.

## Demo

Draw a number between 0 and 9 (the only numbers that exist in this app's universe). The model will tell you what you drew. If you already know what you drew, congratulations, you are smarter than my CNN.

## About

Somewhere around week two of watching YouTube tutorials about convolutional neural networks, I decided the best way to prove I understood deep learning was to train a model on the most beginner dataset in existence, one that has been solved since 1998. The model achieves 99.1% accuracy. I achieve maybe 60% accuracy drawing a clean 4 on the first try. The student has surpassed the master. The master is ashamed.

## Tech Stack

| Layer      | Technology          | Why                                                                                         |
| ---------- | ------------------- | ------------------------------------------------------------------------------------------- |
| Model      | PyTorch (CNN)       | `if pixel == squiggly: return "a number"` didn't get me the internship                     |
| Backend    | Flask               | Spun up an entire HTTP server so you can doodle at it                                       |
| Frontend   | Vanilla HTML/CSS/JS | Tried React. Canvas was not having it. We don't talk about that afternoon.                  |
| Deployment | ONNX Runtime        | Surgically removed PyTorch from production so Vercel doesn't laugh at my requirements.txt  |
| Font       | Roboto Mono         | Absolutely the most critical decision made during this project. Non-negotiable.             |

## Project Structure
```
├── app.py                  # 200 lines of Python that could be replaced by a sticky note
├── convert_to_onnx.py      # Politely asks PyTorch to leave
├── model/
│   ├── mnist_cnn.pth       # 1.2MB of numbers I don't understand
│   └── mnist_cnn.onnx      # Same numbers, now in a bag that Vercel will actually carry
├── notebooks/
│   └── train_model.ipynb   # A document of pure euphoria as the loss curve went down
├── static/
│   ├── index.html          # The one file in this repo I'm genuinely proud of
│   ├── style.css           # Dark theme. I am a very serious engineer.
│   └── app.js              # 300 lines of JavaScript to shuffle pixels into an API call
├── data/                   # 60,000 digits drawn by people in the 90s who had no idea
├── requirements.txt        # A humbling list of people who did the real work
└── .gitignore              # pycache. Always pycache.
```

## Model Architecture

| Layer                        | What It Actually Does        | What I Put In The Report                              |
| ---------------------------- | ---------------------------- | ----------------------------------------------------- |
| Conv2D → BN → ReLU → MaxPool | Looks for edges              | "Hierarchical spatial feature extraction"             |
| Conv2D → BN → ReLU → MaxPool | Looks for more edges         | "Multi-scale abstract representation learning"        |
| Dropout2D (0.25)             | Breaks 25% of neurons, randomly, on purpose | "Stochastic regularization to prevent overfitting" |
| FC → ReLU                    | Multiplies matrices          | "Nonlinear projection into latent decision space"     |
| Dropout (0.5)                | Breaks half of everything, again | "Implicit ensemble learning via noise injection"  |
| FC (128 → 10)                | Picks one of ten numbers     | "Probabilistic multi-class posterior estimation"      |

Six layers. Ten possible outputs. One job. It still gets confused by my 1s.

## Setup

### 1. Clone
```bash
git clone https://github.com/prajxwal/dl-miniproject.git
cd dl-miniproject
# take a moment to appreciate that this is 47 files to classify doodles
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
# pytorch alone is 800MB. pour yourself something.
```

### 3. Train the model (optional, but you'll want to)

Open `notebooks/train_model.ipynb` and run all cells. Watch the loss go down. Feel a warmth in your chest that no other part of this project will replicate. Save the weights. Close the laptop. Go touch grass. You've earned it.

> Skip if `model/mnist_cnn.pth` already exists. It knows what it's doing. Leave it alone.

### 4. Export to ONNX
```bash
python convert_to_onnx.py
```

Strips out PyTorch and repackages the model's entire soul into a single `.onnx` file, because Vercel took one look at my `requirements.txt` and started hyperventilating.

> Skip if `model/mnist_cnn.onnx` exists. Same advice as above.

### 5. Run the app
```bash
python app.py
```

Visit [http://localhost:5000](http://localhost:5000). Draw a digit. Any digit. The model is trying its best. So am I.

## How It Works

1. **You draw** — something vaguely resembling a number on a black canvas
2. **We process** — your artistic vision is immediately cropped, squashed to 28×28 pixels, stripped of all color, and normalized into a float between -1 and 1. Sorry.
3. **The model guesses** — five layers of linear algebra fire in sequence and after approximately 50 milliseconds of fake thinking, produce a number
4. **We celebrate** — animated bars. A big bold digit. A confidence percentage. A footer that says "PyTorch + Flask" so people at job fairs think I know what I'm doing.

## Honest Evaluation

| Scenario                                    | Result                                              |
| ------------------------------------------- | --------------------------------------------------- |
| Clean digit, centered, normal size          | ✅ Correct, every time, insufferably                |
| Test set benchmark                          | ✅ 99.1% — genuinely better than I expected         |
| Digit drawn slightly too small              | ❌ Model stares into the void                       |
| Digit drawn in the corner                   | ❌ Total psychological breakdown                    |
| My handwritten 4                            | ❌ Diagnosed as a 9 with 96% confidence             |
| My handwritten 1                            | ❌ "That's a 7." It is not a 7.                     |

---

*Trained on MNIST · Served with Flask · Deployed with ONNX · Approximately one Wikipedia article's worth of effort dressed up as a portfolio project*