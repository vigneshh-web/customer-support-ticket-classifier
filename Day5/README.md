# Customer Support Ticket Classifier

FastAPI service and fine-tuned DistilBERT models for classifying customer support tickets by category and priority.

## Project Structure

- `Task3.py`: Fine-tunes DistilBERT for category and priority classification, compares performance against a TF-IDF + Logistic Regression baseline, and saves the trained models[cite: 1].
- `Task4.py`: FastAPI application serving `/health`, `/predict`, and `/predict-batch` endpoints alongside an integrated `TestClient` suite[cite: 1].
- `saved_category_model/`: Saved fine-tuned DistilBERT model and tokenizer for ticket category.
- `saved_priority_model/`: Saved fine-tuned DistilBERT model and tokenizer for ticket priority.

## Setup & Installation

1. Install required dependencies:
```bash
pip install torch transformers datasets scikit-learn pandas tabulate fastapi uvicorn pydantic httpx