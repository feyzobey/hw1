# Naive Bayes Classifier

## Overview
This project implements a Naive Bayes Classifier from scratch in Python, without using pre-built machine learning libraries like scikit-learn. The classifier is trained and tested on the "Play Tennis" dataset.

## Dataset
The dataset contains weather-related features (Outlook, Temperature, Humidity, Wind) and a target variable (PlayTennis). The model predicts whether tennis will be played based on weather conditions.

## Features
- Categorical data handling.
- Laplace smoothing for zero probabilities.
- Model persistence in JSON format.
- Leave-One-Out Cross-Validation (LOOCV) for evaluation.

## Requirements
- Python 3.x
- Libraries:
  - pandas
  - numpy
  - (Optional) matplotlib, seaborn for visualization.

Install dependencies with:
```bash
pip install -r requirements.txt
```

Run Program:
```
python main.py
```