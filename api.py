from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load the saved model and vectorizer
model = joblib.load(r'C:\Users\SVI\Desktop\review_sentiments\review_sentiments-main\sentiment_model.pkl')
tfidf = joblib.load(r'C:\Users\SVI\Desktop\review_sentiments\review_sentiments-main\tfidf.pkl')

app = FastAPI()

class ReviewRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: int
    label_name: str

# Map label to sentiment name
label_map = {0: 'negative', 1: 'positive', 2: 'neutral'}

@app.post('/predict', response_model=PredictionResponse)
def predict_sentiment(request: ReviewRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail='Text is required')
    # Preprocess and vectorize
    X = tfidf.transform([request.text.lower()])
    pred = model.predict(X)[0]
    return PredictionResponse(label=int(pred), label_name=label_map.get(int(pred), 'unknown')) 