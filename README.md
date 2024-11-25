# Breast Cancer Prediction Web App and Neural Network Assignment

This repository contains two projects:

1. **Breast Cancer Prediction Web App**: An interactive Streamlit application that predicts whether breast cancer is malignant or benign using a pre-trained Artificial Neural Network (ANN).
2. **Neural Network Assignment**: A Jupyter Notebook showcasing the implementation and evaluation of neural networks for a classification task.

---

## Contents

- [Breast Cancer Prediction Web App](#breast-cancer-prediction-web-app)
  - [Features](#features)
  - [Technologies Used](#technologies-used)
  - [How to Run](#how-to-run)
- [Neural Network Assignment](#neural-network-assignment)
  - [Features](#features-1)
  - [Technologies Used](#technologies-used-1)
  - [How to Use](#how-to-use)
- [License](#license)

---

## Breast Cancer Prediction Web App

### Features

- User-friendly web interface built using Streamlit.
- Accepts 10 key features as input:
  1. Mean Texture
  2. Mean Perimeter
  3. Mean Concavity
  4. Mean Concave Points
  5. Worst Radius
  6. Worst Texture
  7. Worst Perimeter
  8. Worst Area
  9. Worst Concavity
  10. Worst Concave Points
- Pre-processes inputs using a scaler for consistency with training data.
- Predicts the likelihood of the tumor being malignant or benign.
- Displays prediction probabilities for better insights.

### Technologies Used

- Python
- Streamlit
- Joblib (for loading the pre-trained model and scaler)
- Numpy

