from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

MODEL_PATH = "./model/trained"
LABELS = {0: "NEGATIVE", 1: "POSITIVE"}

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

if(torch.backends.mps.is_available()):
    device = "mps"
elif(torch.cuda.is_available()):
    device = "cuda"
else:
    device = "cpu"

device = torch.device(device)
model = model.to(device)
model.eval()  

#####################-

app = FastAPI(title="Sentiment Analyzer API")

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    score: float
    text: str

@app.get("/")
def root():
    return {"status": "ok", "message": "Sentiment Analyzer API"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1)  
    score, predicted_class = torch.max(probs, dim=-1)

    return PredictResponse(
        label=LABELS[predicted_class.item()],
        score=round(score.item(), 4),
        text=request.text
    )

