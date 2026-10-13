# Iris Classifier (Decision Tree)

## Overview

This is a simple machine learning project that uses the data set of Iris to clasify its flowers into three species which are Setosa, Versicolor and Virginica.

## Quick Start 

### 1. Activate the Virtual Environment

On Windows PowerShell:

'''powershell
.\venv\Scripts\Activate.ps1

### 2. Install the required packages

'''powershell
pip install -r requirements.txt

### 3. Run the model

python src/train.py --test-size 0.2 --random-state 42

the program will train the model, print the accuaracy and confusion matrix, save a visual confusion matrix to:
outputs/confusion_matrix.png

## Project Structure 

iris-classifier/
├── data/
├── notebooks/
│  └── iris_model.ipynb
├── outputs/
│  └── confusion_matrix.png
├── src/
│  └── train.py
├── tests/
├── venv/
├── LICENSE
├── README.md
└── requirements.txt

## Model

The project uses a Decision Tree Classifier from scikit-learn.

The iris dataset is split into training and testing data. The model is trained on the training data and evaluated on the test data.

## Results

The model achieved a high accuracy on the test set.

The confusion matrix is saved automatically in the outputs folder.

## License

This project is licensed under the MIT license.


