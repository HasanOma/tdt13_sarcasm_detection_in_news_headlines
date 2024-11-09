# tdt13_sarcasm_detection_in_news_headlines
# Sarcasm Detection in News Headlines

This project explores sarcasm detection in news headlines using three different models:

- **Logistic Regression** with TF-IDF features
- **LSTM-based Recurrent Neural Network**
- **Transformer-based model using DistilBERT**

## Table of Contents

- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Training the Models](#training-the-models)
- [Running the Flask App](#running-the-flask-app)

## Project Overview

The goal of this project is to develop and compare different models for detecting sarcasm in news headlines. The models are trained on the "News Headlines Dataset for Sarcasm Detection," which contains labeled headlines from **The Onion** and **HuffPost**. The repository of the dataset can be found [here](https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection/tree/master)

## Prerequisites

- Python 3.7 or higher
- Git

## Installation

Follow these steps to set up the project on your local machine.

### 1. Clone the Repository

```bash
    git clone https://github.com/HasanOma/tdt13_sarcasm_detection_in_news_headlines.git
    cd tdt13_sarcasm_detection_in_news_headlines
```

### 2. Create a Virtual Environment
It's recommended to use a virtual environment to manage dependencies.

Using venv
```bash
    python -m venv venv
```
Activate the Virtual Environment
On Windows:
```bash
    venv\Scripts\activate
```
On macOS/Linux:
```bash
    source venv/bin/activate
```

### 3. Install Dependencies

```bash
    pip install -r requirements.txt
```

### Training the Models
Before running the Flask app, you need to train the models and generate the necessary files.

#### 1. Train the Models
The training script is designed to train all three models. You can specify which model to train by setting the MODEL_TYPE variable.

Open the Training Script
Open the sarcasmdetection.py script in your preferred text editor.

Set the Model Type
Uncomment or set the MODEL_TYPE variable to the desired model:
```bash
    MODEL_TYPE = 'lstm'  # Options: 'distilbert', 'lstm', 'logistic_regression'
```
Run the Training Script
```bash
    python sarcasmdetection.py
```
Repeat the above steps for each model by changing the MODEL_TYPE variable accordingly.

### Training Outputs

#### Logistic Regression:
logistic_regression_model.pkl
tfidf_vectorizer.pkl

#### LSTM Model:

lstm_model_state.bin

vocab.pkl

### DistilBERT Model:

best_model_state.bin

tokenizer.pickle

Paste the desired models into the Results Folder.

## Running the Flask App
Now that the models are trained and placed in the results folder, you can run the Flask app to interact with the sarcasm detection models.

### 1. Ensure the Virtual Environment is Activated
If you haven't already activated your virtual environment, do so now.

On Windows:
```bash
    venv\Scripts\activate
```

On macOS/Linux:
```bash
    source venv/bin/activate
```

#### Install requirements
Now that you are in the virtual environment, install the needed dependencies in the requirements.txt file.
```bash
    pip install -r requirements.txt
```

### 2. Run the application
```bash
    python app.py
```
The app should now be running on http://127.0.0.1:5000/.