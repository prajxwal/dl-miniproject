# MNIST Digit Recognizer

A neural network that recognizes handwritten digits. A kindergartener can do this for free. I spent three weeks on it anyway.

## Demo

Draw a number between 0 and 9 (the only numbers that exist in this app's universe). The model will tell you what you drew. If you draw something that is clearly not a number, the model will now tell you *that*, too — which is more self-awareness than I can claim most days.

## About

Somewhere around week two of watching YouTube tutorials about convolutional neural networks, I decided the best way to prove I understood deep learning was to train a model on the most beginner dataset in existence, one that has been solved since 1998. The model achieves 99.25% accuracy across 11 classes. I achieve maybe 60% accuracy drawing a clean 4 on the first try. The student has surpassed the master. The master is ashamed.

The model originally had no concept of "that's not a number" and would confidently classify your cat doodle as a 7. We fixed that by teaching it an 11th class: *Junk*. Six thousand synthetic images of random scribbles later, the model can now say "I have no idea what that is" — a milestone that took me considerably longer to reach in my own life.

## Tech Stack

| Layer      | Technology          | Why                                                                                         |
| ---------- | ------------------- | ------------------------------------------------------------------------------------------- |
| Model      | PyTorch (CNN)       | `if pixel == squiggly: return "a number"` didn't get me the internship                     |
| Backend    | Flask               | Spun up an entire HTTP server so you can doodle at it                                       |
| Frontend   | Vanilla HTML/CSS/JS | Tried React. Canvas was not having it. We don't talk about that afternoon.                  |
| Deployment | ONNX Runtime        | Surgically removed PyTorch from production so Vercel doesn't laugh at my requirements.txt  |
| Junk Class | Pillow + randomness | 6,000 images of lines going nowhere. Relatable.                                            |
| Font       | Roboto Mono         | Absolutely the most critical decision made during this project. Non-negotiable.             |

## Project Structure
```
├── app.py                  # 130 lines of Python pretending to be a production server
├── model/
│   ├── mnist_cnn.pth       # 1.6MB of numbers I don't understand
│   └── mnist_cnn.onnx      # Same numbers, now in a bag that Vercel will actually carry
├── notebooks/
│   └── train_model.ipynb   # One notebook to rule them all. Training, junk synthesis,
│                           #   visualization, confusion matrix, and ONNX export.
│                           #   Previously three files. We had a refactoring intervention.
├── static/
│   ├── index.html          # The one file in this repo I'm genuinely proud of
│   ├── style.css           # Dark theme. I am a very serious engineer.
│   └── app.js              # 340 lines of JavaScript to shuffle pixels into an API call
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
| FC (128 → 11)                | Picks one of eleven options  | "Probabilistic multi-class posterior estimation"      |

Six layers. Eleven possible outputs. One of them is "I don't know." Honestly, same.

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

Open `notebooks/train_model.ipynb` and run all cells. Watch the loss go down. Feel a warmth in your chest that no other part of this project will replicate. The notebook trains on MNIST digits *and* generates 6,000 synthetic junk images so the model learns to reject your abstract art. It also exports the ONNX model at the end — one notebook does everything now because I got tired of maintaining three files that all defined the same class.

> Skip if `model/mnist_cnn.onnx` already exists. It knows what it's doing. Leave it alone.

### 4. Run the app
```bash
python app.py
```

Visit [http://localhost:5000](http://localhost:5000). Draw a digit. Any digit. The model is trying its best. So am I.

## How It Works

1. **You draw** — something vaguely resembling a number on a black canvas
2. **We process** — your artistic vision is immediately cropped, squashed to 28×28 pixels, stripped of all color, and normalized into a float between -1 and 1. Sorry.
3. **The model guesses** — six layers of linear algebra fire in sequence and after approximately 50 milliseconds of fake thinking, produce one of eleven answers (ten digits, or "that's not a digit, what are you doing")
4. **We celebrate** — animated bars. A big bold digit. A confidence percentage. Or a big bold question mark, which is the model's way of saying "please draw an actual number."

## Honest Evaluation

| Scenario                                    | Result                                              |
| ------------------------------------------- | --------------------------------------------------- |
| Clean digit, centered, normal size          | ✅ Correct, every time, insufferably                |
| Test set benchmark                          | ✅ 99.25% — genuinely better than I expected        |
| Random squiggly lines                       | ✅ "Not a digit" — finally                         |
| Digit drawn slightly too small              | ❌ Model stares into the void                       |
| Digit drawn in the corner                   | ❌ Total psychological breakdown                    |
| My handwritten 4                            | ❌ Diagnosed as a 9 with 96% confidence             |
| My handwritten 1                            | ❌ "That's a 7." It is not a 7.                     |

---

*Trained on MNIST · Augmented with synthetic junk · Served with Flask · Deployed with ONNX · Approximately one Wikipedia article's worth of effort dressed up as a portfolio project*