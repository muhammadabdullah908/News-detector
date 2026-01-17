# Fake News Detector

A Python-based application to classify news articles as **Real** or **Fake** using **NLP (TF-IDF + Logistic Regression)**. The application supports both **command-line training** and a **GUI interface** for testing articles interactively.

---

## Features

* Train a Logistic Regression model on the **Kaggle Fake & True News dataset**.
* Classify news articles using **title + text**.
* **GUI interface** using Tkinter:

  * Input title and text manually
  * Predict article as **REAL** or **FAKE**
  * Load a **random sample** from the dataset for testing
  * Clear fields for new input
* Save and load trained model to avoid retraining every time.
## fake news detection AI project 2025 3rd sem
---

## Dataset

* Kaggle’s [Fake & Real News Dataset](https://www.kaggle.com/datasets/nileshbhate/fake-and-real-news-dataset)
* Files used:

  * `Fake.csv` → labeled as **FAKE**
  * `True.csv` → labeled as **REAL**
* Columns used: `title`, `text`, `label`

---

## Installation

1. Clone or download this repository.
2. Make sure you have Python 3.8+ installed.
3. Install required packages:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Training the Model (Optional)

The `main.py` automatically trains the model if `models/fakenews_pipeline.joblib` does not exist. You can also force retraining by deleting the `models` folder.

```bash
python main.py
```

During training:

* The Kaggle dataset (`Fake.csv` + `True.csv`) is loaded
* Labels are assigned: `0 = Real`, `1 = Fake`
* TF-IDF + Logistic Regression pipeline is trained
* Model is saved as `models/fakenews_pipeline.joblib`

---

### 2. Using the GUI

Run the application:

```bash
python main.py
```

* **Title**: Enter the article’s title
* **Text**: Enter the article content
* **Predict**: Shows prediction and probabilities
* **Load Random Sample**: Fills fields with a random article from the dataset
* **Clear Fields**: Clears input/output fields

Example GUI layout:

```
+-----------------------------+
| Title                       |
| [Input field]               |
+-----------------------------+
| Text                        |
| [Text field]                |
+-----------------------------+
| Prediction Output           |
| [REAL / FAKE + Probabilities]|
+-----------------------------+
| [Predict] [Load Sample] [Clear] |
+-----------------------------+
```

---

## Model Details

* **Vectorizer**: TF-IDF (removes English stopwords, ignores terms appearing in >70% of documents)
* **Classifier**: Logistic Regression (max_iter=1000)
* Trained on combined Kaggle dataset (`Fake.csv` + `True.csv`)
* Model saved at `models/fakenews_pipeline.joblib` for reuse

---

## Example Prediction

* **Input Title**: "Government Approves New Health Policy"
* **Input Text**: "The government has passed a new law to expand access to healthcare in rural regions."
* **Prediction Output**:

```
Prediction: REAL
Probability REAL: 0.9123
Probability FAKE: 0.0877
```

---

## Requirements (`requirements.txt`)

```
pandas
scikit-learn
tk
joblib
```

---

## Notes

* Make sure `Fake.csv` and `True.csv` are in the same directory as `main.py`.
* The GUI uses Tkinter; no additional installation required for standard Python distributions.
* The dataset is **Kaggle licensed**; please respect its usage terms.
