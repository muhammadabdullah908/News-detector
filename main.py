import tkinter as tk
from tkinter import Text, Entry, Label, Button, END, messagebox
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import random
import os

# ---------------- CONFIG ----------------
MODEL_PATH = "models/fakenews_pipeline.joblib"
FAKE_CSV = "Fake.csv"
TRUE_CSV = "True.csv"

# ---------------- LOAD DATA ----------------  this is my changes
#
def load_kaggle_dataset(fake_path, true_path):
    try:
        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load dataset: {e}")
        return None

    # Assign labels: 0 = Real, 1 = Fake
    fake_df['label'] = 1
    true_df['label'] = 0

    # Only keep 'title' and 'text' columns
    fake_df = fake_df[['title', 'text', 'label']]
    true_df = true_df[['title', 'text', 'label']]

    # Combine and shuffle
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

df = load_kaggle_dataset(FAKE_CSV, TRUE_CSV)

# ---------------- TRAIN MODEL ----------------
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    if df is None:
        messagebox.showerror("Error", "No dataset available to train model!")
        exit()
    # Train TF-IDF + Logistic Regression pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    print("Training model on Kaggle dataset ...")
    X = df['title'] + " " + df['text']
    y = df['label']
    pipeline.fit(X, y)
    model = pipeline
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Model trained and saved to", MODEL_PATH)

# ---------------- GUI ----------------
root = tk.Tk()
root.title("Fake News Detector")
root.geometry("800x650")

# Title
Label(root, text="Title").pack()
title_entry = Entry(root, width=100)
title_entry.pack()

# Text
Label(root, text="Text").pack()
text_entry = Text(root, height=15, width=95)
text_entry.pack()

# Output
output = Text(root, height=12, width=95)
output.pack()

# ---------------- FUNCTIONS ----------------
def predict_article():
    title = title_entry.get()
    text = text_entry.get("1.0", END).strip()
    if not title and not text:
        messagebox.showwarning("Input Missing", "Please enter title or text.")
        return

    combined = title + "\n" + text
    pred = model.predict([combined])[0]
    prob = model.predict_proba([combined])[0]
    label = "FAKE" if pred == 1 else "REAL"

    output.delete("1.0", END)
    output.insert(END, f"Prediction: {label}\n")
    output.insert(END, f"Probability REAL: {prob[0]:.4f}\n")
    output.insert(END, f"Probability FAKE: {prob[1]:.4f}\n")


#changed code

def load_random_sample():
    """
    Load a random news article from the dataset
    and populate the Title and Text fields.
    """
    if df is None or df.empty:
        messagebox.showerror("Dataset Missing", "Dataset is not available or empty.")
        return

    # Select a random row
    random_row = df.sample(n=1).iloc[0]

    # Clear existing fields
    title_entry.delete(0, END)
    text_entry.delete("1.0", END)
    output.delete("1.0", END)

    # Insert new data
    title_entry.insert(END, str(random_row["title"]))
    text_entry.insert("1.0", str(random_row["text"]))

    # Inform user
    output.insert(END, "Random sample loaded successfully.\n")


def clear_all_fields():
    """Clear title, text, and output fields."""
    title_entry.delete(0, END)
    text_entry.delete("1.0", END)
    output.delete("1.0", END)


# ---------------- BUTTONS ----------------
Button(
    root,
    text="Predict",
    command=predict_article,
    width=25,
    bg="lightblue"
).pack(pady=6)

Button(
    root,
    text="Load Random News",
    command=load_random_sample,
    width=25,
    bg="lightgreen"
).pack(pady=6)

Button(
    root,
    text="Clear All",
    command=clear_all_fields,
    width=25,
    bg="lightgray"
).pack(pady=6)

# ---------------- MAIN LOOP ----------------
root.mainloop()
