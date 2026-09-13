# SignVision AI

A graduate computer-vision project that classifies static images of American Sign
Language (ASL) alphabet gestures and studies how the classifier behaves under
adversarial attack. Built with TensorFlow/Keras and presented as an interactive
Streamlit app.

> **Scope:** This is static-image classification of the ASL *alphabet* (26 letters
> plus three control gestures), not real-time or continuous sign-language
> translation.

**Live demo:** [signvision-ai.streamlit.app](https://signvision-ai.streamlit.app/)

---

## What this project does

1. Train a baseline image classifier on the ASL alphabet.
2. Attack it with the Fast Gradient Sign Method (FGSM) to measure how easily its
   predictions can be flipped by small input perturbations.
3. Retrain the model on adversarial examples and compare the robustness
   trade-off against the baseline.

It began as my final project for CMSE 890 (Computational Mathematics, Science &
Engineering) at Michigan State University.

---

## Model

- **Architecture:** VGG16 convolutional base (from `keras.applications`) with a
  flatten layer and a softmax classification head.
- **Input:** 50×50 RGB images, scaled to `[0, 1]`.
- **Classes:** 29 (the letters A–Z plus `delete`, `nothing`, and `space`).
- **Framework:** TensorFlow 2.19 / Keras 3.

> **Note:** the trained weight file exposes 30 output units; the 30th is an
> unused artifact of training and is never surfaced by the app. See
> [`MODEL_CARD.md`](MODEL_CARD.md) for details.

---

## Results

- **Baseline test accuracy:** ~93.6% on clean images (from `results/logs.npy`).
- **Adversarial vulnerability:** FGSM achieves a high misclassification rate
  against the baseline model (~93% attack success across the evaluated set).
- **After adversarial training:** natural accuracy decreases modestly while
  accuracy on adversarial inputs improves substantially. This is the expected
  robustness trade-off.

The robustness figures shown in the app come from the adversarial-training
experiments run for the CMSE 890 project.

---

## Dataset

The [ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet):
~87,000 labeled images (roughly 3,000 per class) across 29 classes, captured
against a uniform background.

---

## Project structure

```
signvision-ai/
├── streamlit_app/
│   ├── Home.py              # Landing page
│   ├── config.py            # Shared paths, class labels, cached model loader
│   ├── pages/               # Classify, Adversarial, Results, Bio
│   └── assets/              # Model weights, images, logos, fonts
├── src/
│   ├── fit.py               # Model construction & training helpers
│   ├── evaluate.py          # Evaluation helper
│   ├── viz.py               # Plotting (confusion matrix, learning curves)
│   ├── model_fix.py         # Inspect saved weight shapes
│   ├── model.ipynb          # Baseline training experiment
│   ├── advML.ipynb          # Adversarial attacks & adversarial training
│   └── metrics.ipynb        # Evaluation & metrics
├── results/                 # logs.npy, confusion_matrix.csv, distPlot.html
├── docs/                    # Project proposal & final presentation (PDF)
├── requirements.txt
├── runtime.txt              # python-3.10.12
└── README.md
```

---

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app/Home.py
```

Requires Python 3.10.

---

## Tech stack

- **ML:** TensorFlow, Keras, Foolbox (adversarial attacks), scikit-learn
- **Data/plots:** NumPy, pandas, Matplotlib, seaborn, OpenCV
- **App:** Streamlit

---

## Limitations

- Classifies single static images, not video or continuous signing.
- Trained on a single, uniformly-lit dataset, so accuracy on real-world photos
  (varied lighting, backgrounds, skin tones, hand positions) will be lower.
- This is a research and portfolio project, not a production communication aid.

---

For questions or feedback, contact Mikayla Norton at
[mikayla.e.norton@gmail.com](mailto:mikayla.e.norton@gmail.com).
