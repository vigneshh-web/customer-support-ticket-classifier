from contextlib import asynccontextmanager
from typing import List
import sys
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field, field_validator
from transformers import AutoTokenizer, AutoModelForSequenceClassification

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models["device"] = device
    models["tokenizer"] = AutoTokenizer.from_pretrained("./saved_category_model")
    
    cat_model = AutoModelForSequenceClassification.from_pretrained("./saved_category_model")
    cat_model.to(device)
    cat_model.eval()
    models["category_model"] = cat_model
    
    prio_model = AutoModelForSequenceClassification.from_pretrained("./saved_priority_model")
    prio_model.to(device)
    prio_model.eval()
    models["priority_model"] = prio_model
    
    yield
    models.clear()

app = FastAPI(title="Customer Support Ticket Classifier", lifespan=lifespan)

class TicketRequest(BaseModel):
    text: str = Field(..., min_length=1)

    @field_validator("text")
    @classmethod
    def check_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace")
        return v

class TicketPrediction(BaseModel):
    category: str
    priority: str
    confidence: float

class BatchTicketRequest(BaseModel):
    tickets: List[TicketRequest] = Field(..., min_length=1)

class BatchTicketResponse(BaseModel):
    predictions: List[TicketPrediction]

def run_inference(texts: List[str]) -> List[TicketPrediction]:
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    tokenizer = models["tokenizer"]
    cat_model = models["category_model"]
    prio_model = models["priority_model"]
    device = models["device"]
    
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        cat_outputs = cat_model(**inputs)
        cat_probs = F.softmax(cat_outputs.logits, dim=-1)
        cat_confs, cat_preds = torch.max(cat_probs, dim=-1)
        
        prio_outputs = prio_model(**inputs)
        prio_probs = F.softmax(prio_outputs.logits, dim=-1)
        _, prio_preds = torch.max(prio_probs, dim=-1)
        
    results = []
    for i in range(len(texts)):
        category = cat_model.config.id2label[cat_preds[i].item()]
        priority = prio_model.config.id2label[prio_preds[i].item()]
        confidence = round(cat_confs[i].item(), 2)
        results.append(
            TicketPrediction(
                category=category,
                priority=priority,
                confidence=confidence
            )
        )
    return results

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": "category_model" in models and "priority_model" in models
    }

@app.post("/predict", response_model=TicketPrediction)
def predict_ticket(ticket: TicketRequest):
    return run_inference([ticket.text])[0]

@app.post("/predict-batch", response_model=BatchTicketResponse)
def predict_batch(batch: BatchTicketRequest):
    texts = [t.text for t in batch.tickets]
    predictions = run_inference(texts)
    return BatchTicketResponse(predictions=predictions)

def run_tests():
    with TestClient(app) as client:
        res_health = client.get("/health")
        assert res_health.status_code == 200
        assert res_health.json() == {"status": "healthy", "models_loaded": True}

        res_valid = client.post("/predict", json={"text": "I was charged twice"})
        assert res_valid.status_code == 200
        data = res_valid.json()
        assert "category" in data and "priority" in data and "confidence" in data
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0

        res_empty = client.post("/predict", json={"text": ""})
        assert res_empty.status_code == 422

        res_blank = client.post("/predict", json={"text": "   "})
        assert res_blank.status_code == 422

        res_missing = client.post("/predict", json={})
        assert res_missing.status_code == 422

        res_batch = client.post("/predict-batch", json={
            "tickets": [
                {"text": "I was charged twice"},
                {"text": "App crashes on login"}
            ]
        })
        assert res_batch.status_code == 200
        batch_data = res_batch.json()
        assert len(batch_data["predictions"]) == 2

    print("All tests passed successfully!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    else:
        uvicorn.run(app, host="127.0.0.1", port=8000)