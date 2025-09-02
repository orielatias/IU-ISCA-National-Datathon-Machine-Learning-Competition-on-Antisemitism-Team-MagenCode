import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# Load the gold standard dataset
df = pd.read_csv("GoldStandard2024.csv")

# Preview the columns to confirm structure
print(df.columns)

# Ensure the columns are named 'text' and 'label'
df = df.rename(columns={"Text": "text", "Biased": "label"})

# Drop NaNs and filter unexpected labels
df = df.dropna(subset=["text", "label"])
df = df[df["label"].isin([0, 1])]  # Only keep binary labels

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Train/test split
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# Tokenize using BERTweet tokenizer
model_name = "vinai/bertweet-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load model and trainer setup
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()
