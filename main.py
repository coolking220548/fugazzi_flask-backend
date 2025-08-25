from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load tokenizer + model
tokenizer = DistilBertTokenizer.from_pretrained("./distilBERT")
model = DistilBertForSequenceClassification.from_pretrained("./distilBERT")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewsItem(BaseModel):
    text: str

@app.post("/detect")
def predict(request: NewsItem):
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True, max_length=256)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    prediction = int(probs.argmax())
    confidence = float(probs.max() * 100)

    return {
        "classification": "True" if prediction == 1 else "False",
        "explanation": f"Model confidence is {confidence:.2f}%"
    }
