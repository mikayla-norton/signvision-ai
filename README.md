# SignVision AI

A portfolio project showcasing adversarial-attack experiments on sign-language classification models, built with Keras and TensorFlow. This private reference directs viewers to a live Streamlit demonstration of the application.

---

## Overview

SignVision AI evaluates the robustness of sign-language image classifiers under adversarial perturbations. Key objectives:

* Assess model vulnerability to adversarial attacks (FGSM, PGD, DeepFool).
* Quantify degradation in classification accuracy on perturbed inputs.
* Visualize adversarial examples alongside clean images.

---

## Live Demo

Access the interactive Streamlit application to explore the model’s performance and adversarial examples:

[SignVision AI Streamlit App](https://signvision-ai.streamlit.app/)

---

## Architecture

The pipeline consists of:

1. **Data Input**: Preprocessed sign-language image dataset.
2. **Model**: Convolutional neural network implemented in Keras, based on a ResNet-18 backbone.
3. **Adversarial Attacks**:

   * Fast Gradient Sign Method (FGSM)
   * Projected Gradient Descent (PGD)
   * DeepFool
4. **Evaluation**: Compute accuracy on clean vs. adversarial sets; generate confusion matrices.
5. **Visualization**: Plot examples of adversarial perturbations and corresponding model predictions.

---

## Model Performance Summary

* **Baseline Test Accuracy:** 93.6%
* **Robust Natural Accuracy:** 84%
* **Robust Adversarial Accuracy:** 97%

The adversarially trained model shows a trade-off: a slight decrease in natural accuracy in exchange for substantial robustness improvements against adversarial examples.

---

## Project Structure

```
signvision-ai/
├── streamlit_app/    # Streamlit demo code and multipage scripts
├── src/              # Core modules: model definitions, attack implementations, utilities
├── notebooks/        # Jupyter notebooks for analysis and visualization
├── data/             # Small sample datasets (e.g., confusion matrices)
├── results/          # Generated artifacts: plots, logs, HTML outputs
├── docs/             # Documentation and proposal PDFs
├── .gitignore        # Git ignore rules for temporary and large files
├── requirements.txt  # List of Python dependencies
└── README.md         # Project overview and instructions
```

---

## Technologies

* **Frameworks**: TensorFlow, Keras
* **Languages**: Python 3.8+
* **Libraries**: NumPy, SciPy, Matplotlib
* **Hosting**: Streamlit

---

For questions or feedback, please contact Mikayla Norton at [mikayla.e.norton@gmail.com](mailto:mikayla.e.norton@gmail.com).
