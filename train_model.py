import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
import json

# ===============================
# Step 1: Load dataset
# ===============================
print("Loading dataset...")

# Try both separators automatically
try:
    data = pd.read_csv("my_dataset.csv", on_bad_lines="skip", engine="python")
    if len(data.columns) == 1:
        data = pd.read_csv("my_dataset.csv", sep="\t", on_bad_lines="skip", engine="python")
except Exception as e:
    raise RuntimeError(f"Error reading dataset: {e}")

print(f"Loaded {len(data)} rows.")
print(f"Columns detected: {list(data.columns)}\n")

# Try to fix wrong column parsing
if len(data.columns) == 1 and ',' in data.columns[0]:
    # Split the single-column header into multiple columns
    data = data.columns[0].split(',')
    print("⚠️ Detected broken CSV header — please check your file formatting!")
    exit()

# --- Normalize column names ---
cols_lower = [c.lower().strip() for c in data.columns]
data.columns = cols_lower

# Try to find text and label columns dynamically
text_col = None
label_col = None

for c in cols_lower:
    if "text" in c or "comment" in c or "tweet" in c or "content" in c:
        text_col = c
    if "label" in c or "category" in c or "class" in c:
        label_col = c

if not text_col or not label_col:
    raise ValueError(f"❌ Could not find proper text/label columns in {list(data.columns)}")

print(f"✅ Using '{text_col}' as text column and '{label_col}' as label column\n")

# ===============================
# Step 1.5: Clean and filter labels
# ===============================
label_map = {"safe": 0, "offensive": 1, "hate_speech": 2}

# Normalize label text
data[label_col] = data[label_col].astype(str).str.lower().str.strip()

# Remove invalid labels
valid_labels = set(label_map.keys())
before_rows = len(data)
data = data[data[label_col].isin(valid_labels)]
after_rows = len(data)
print(f"Filtered out {before_rows - after_rows} invalid-labeled rows. Remaining: {after_rows}")

# Map text labels to numeric
data[label_col] = data[label_col].map(label_map)

# Drop missing
data = data.dropna(subset=[text_col, label_col])

print("✅ Labels cleaned and mapped successfully.\n")

# ===============================
# Step 2: Convert to Hugging Face Dataset
# ===============================
dataset = Dataset.from_dict({
    "text": data[text_col].astype(str).tolist(),
    "label": data[label_col].astype(int).tolist()
})

print("Converted to Hugging Face Dataset.\n")

# ===============================
# Step 3: Tokenization
# ===============================
tokenizer = BertTokenizer.from_pretrained("google/muril-base-cased")

def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

encoded_dataset = dataset.map(preprocess_function, batched=True)
print("Data preprocessing complete.\n")

# ===============================
# Step 4: Train/Test Split
# ===============================
train_test_split = encoded_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]
print("Data split into training and testing sets successfully!\n")

# ===============================
# Step 5: Model Setup
# ===============================
model = BertForSequenceClassification.from_pretrained(
    "google/muril-base-cased",
    num_labels=3
)

# ===============================
# Step 6: Training Arguments
# ===============================
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    save_safetensors=False
)

# ===============================
# Step 7: Trainer
# ===============================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    accuracy = (preds == labels).astype(float).mean().item()
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ===============================
# Step 8: Train
# ===============================
print("Starting model training...")
trainer.train()

# ===============================
# Step 9: Save Model, Tokenizer, and Label Map
# ===============================
os.makedirs("./trained_model", exist_ok=True)
print("Saving model and tokenizer...")
model.save_pretrained("./trained_model", safe_serialization=False)
tokenizer.save_pretrained("./trained_model")

with open("./trained_model/label_map.json", "w") as f:
    json.dump(label_map, f)

print("🎉 Training complete! Model and tokenizer saved in ./trained_model/")
