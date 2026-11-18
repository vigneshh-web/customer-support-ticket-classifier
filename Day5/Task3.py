import time
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

data = {
    "text": [
        "I was charged twice for my monthly subscription",
        "My payment failed but money was deducted from bank",
        "Need an invoice for last month's payment",
        "How do I update my credit card details?",
        "Requesting a refund for accidental renewal",
        "App crashes immediately when I try to login",
        "Getting a 500 internal server error on dashboard",
        "Two-factor authentication code is not arriving",
        "Cannot export my reports to CSV format",
        "Website is loading very slowly since the update",
        "How do I change my account password?",
        "Where can I find the user guide for API setup?",
        "Do you offer discounts for non-profit organizations?",
        "I want to upgrade my plan to Enterprise",
        "How many team members can I add to my workspace?",
        "Please delete my account and all associated data",
        "Change the primary email address on my profile",
        "Billing cycle shows wrong renewal date",
        "Dark mode toggle is broken on mobile app",
        "Cancel my subscription effective immediately"
    ] * 10,
    "category": [
        "Billing", "Billing", "Billing", "Billing", "Billing",
        "Technical", "Technical", "Technical", "Technical", "Technical",
        "General", "General", "General", "General", "General",
        "Account", "Account", "Billing", "Technical", "Account"
    ] * 10,
    "priority": [
        "High", "High", "Low", "Medium", "High",
        "High", "High", "High", "Medium", "Medium",
        "Low", "Low", "Low", "Medium", "Low",
        "Medium", "Low", "Medium", "Low", "High"
    ] * 10
}

df = pd.DataFrame(data)

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["category"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["category"])

start_baseline_train = time.time()
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(train_df["text"])
X_val_tfidf = vectorizer.transform(val_df["text"])
X_test_tfidf = vectorizer.transform(test_df["text"])

baseline_clf = LogisticRegression(max_iter=1000, random_state=42)
baseline_clf.fit(X_train_tfidf, train_df["category"])
baseline_train_time = time.time() - start_baseline_train

baseline_val_preds = baseline_clf.predict(X_val_tfidf)
baseline_val_acc = accuracy_score(val_df["category"], baseline_val_preds)
baseline_val_f1 = f1_score(val_df["category"], baseline_val_preds, average="macro")

start_baseline_inf = time.time()
for text in test_df["text"]:
    vec = vectorizer.transform([text])
    _ = baseline_clf.predict(vec)
baseline_inf_time_ms = ((time.time() - start_baseline_inf) / len(test_df)) * 1000

baseline_test_preds = baseline_clf.predict(X_test_tfidf)
baseline_test_acc = accuracy_score(test_df["category"], baseline_test_preds)
baseline_test_f1 = f1_score(test_df["category"], baseline_test_preds, average="macro")

categories = sorted(df["category"].unique())
label2id = {label: idx for idx, label in enumerate(categories)}
id2label = {idx: label for idx, label in enumerate(categories)}

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["label"] = train_df["category"].map(label2id)
val_df["label"] = val_df["category"].map(label2id)
test_df["label"] = test_df["category"].map(label2id)

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

train_dataset = Dataset.from_pandas(train_df[["text", "label"]]).map(tokenize_function, batched=True)
val_dataset = Dataset.from_pandas(val_df[["text", "label"]]).map(tokenize_function, batched=True)
test_dataset = Dataset.from_pandas(test_df[["text", "label"]]).map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(categories),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./distilbert_category_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_steps=10,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

start_bert_train = time.time()
trainer.train()
bert_train_time = time.time() - start_bert_train

bert_val_metrics = trainer.evaluate(val_dataset)
bert_test_metrics = trainer.evaluate(test_dataset)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

start_bert_inf = time.time()
with torch.no_grad():
    for text in test_df["text"]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        outputs = model(**inputs)
        _ = torch.argmax(outputs.logits, dim=-1).item()
bert_inf_time_ms = ((time.time() - start_bert_inf) / len(test_df)) * 1000

priorities = sorted(df["priority"].unique())
prio_label2id = {label: idx for idx, label in enumerate(priorities)}
prio_id2label = {idx: label for idx, label in enumerate(priorities)}

train_df["prio_label"] = train_df["priority"].map(prio_label2id)
val_df["prio_label"] = val_df["priority"].map(prio_label2id)
test_df["prio_label"] = test_df["priority"].map(prio_label2id)

train_prio_ds = Dataset.from_pandas(train_df[["text", "prio_label"]].rename(columns={"prio_label": "label"})).map(tokenize_function, batched=True)
val_prio_ds = Dataset.from_pandas(val_df[["text", "prio_label"]].rename(columns={"prio_label": "label"})).map(tokenize_function, batched=True)
test_prio_ds = Dataset.from_pandas(test_df[["text", "prio_label"]].rename(columns={"prio_label": "label"})).map(tokenize_function, batched=True)

prio_model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(priorities),
    id2label=prio_id2label,
    label2id=prio_label2id
)

prio_training_args = TrainingArguments(
    output_dir="./distilbert_priority_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_steps=10,
    report_to="none"
)

prio_trainer = Trainer(
    model=prio_model,
    args=prio_training_args,
    train_dataset=train_prio_ds,
    eval_dataset=val_prio_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

start_prio_train = time.time()
prio_trainer.train()
prio_train_time = time.time() - start_prio_train

prio_val_metrics = prio_trainer.evaluate(val_prio_ds)
prio_test_metrics = prio_trainer.evaluate(test_prio_ds)

prio_model.to(device)
prio_model.eval()

start_prio_inf = time.time()
with torch.no_grad():
    for text in test_df["text"]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        outputs = prio_model(**inputs)
        _ = torch.argmax(outputs.logits, dim=-1).item()
prio_inf_time_ms = ((time.time() - start_prio_inf) / len(test_df)) * 1000

print("\n--- VALIDATION SET METRICS ---")
print(f"Baseline (TF-IDF + LR) -> Accuracy: {baseline_val_acc:.4f} | Macro F1: {baseline_val_f1:.4f}")
print(f"DistilBERT (Category)  -> Accuracy: {bert_val_metrics['eval_accuracy']:.4f} | Macro F1: {bert_val_metrics['eval_macro_f1']:.4f}")
print(f"DistilBERT (Priority)  -> Accuracy: {prio_val_metrics['eval_accuracy']:.4f} | Macro F1: {prio_val_metrics['eval_macro_f1']:.4f}")

comparison_df = pd.DataFrame([
    {
        "Model": "Baseline (TF-IDF + Logistic Regression)",
        "Accuracy": f"{baseline_test_acc:.4f}",
        "Macro F1": f"{baseline_test_f1:.4f}",
        "Training time": f"{baseline_train_time:.2f}s",
        "Inference time per ticket": f"{baseline_inf_time_ms:.2f}ms"
    },
    {
        "Model": "Fine-tuned DistilBERT (Category)",
        "Accuracy": f"{bert_test_metrics['eval_accuracy']:.4f}",
        "Macro F1": f"{bert_test_metrics['eval_macro_f1']:.4f}",
        "Training time": f"{bert_train_time:.2f}s",
        "Inference time per ticket": f"{bert_inf_time_ms:.2f}ms"
    },
    {
        "Model": "Fine-tuned DistilBERT (Priority)",
        "Accuracy": f"{prio_test_metrics['eval_accuracy']:.4f}",
        "Macro F1": f"{prio_test_metrics['eval_macro_f1']:.4f}",
        "Training time": f"{prio_train_time:.2f}s",
        "Inference time per ticket": f"{prio_inf_time_ms:.2f}ms"
    }
])

print("\n--- TEST SET COMPARISON TABLE ---")
print(comparison_df.to_string(index=False))

model.save_pretrained("./saved_category_model")
tokenizer.save_pretrained("./saved_category_model")
prio_model.save_pretrained("./saved_priority_model")
tokenizer.save_pretrained("./saved_priority_model")