# Sentiment Analysis on Product Reviews

This project performs sentiment analysis on product reviews using machine learning and deep learning models. It includes data preprocessing, model training (Logistic Regression, Naive Bayes, Decision Tree, Random Forest), and exposes a prediction API using FastAPI, as well as a user-friendly interface using Streamlit.

## Features
- Data cleaning and preprocessing (punctuation, stopwords, lemmatization, emoji removal)
- Model training and evaluation (Logistic Regression, Naive Bayes, Decision Tree, Random Forest)
- Model comparison using LazyClassifier
- Balanced dataset option for fair training
- FastAPI backend for real-time predictions
- Streamlit frontend for easy user interaction

## Project Structure
```
review_sentiments-main/
├── review.ipynb           # Main Jupyter notebook for data processing and model training
├── review1.csv, ...       # Raw review datasets
├── api.py                 # FastAPI app for serving predictions
├── streamlit_app.py       # Streamlit app for user interface
├── sentiment_model.pkl    # Saved ML model (after training)
├── tfidf.pkl              # Saved TF-IDF vectorizer (after training)
├── README.md              # This file
```

## Requirements
- Python 3.7+
- pandas, numpy, scikit-learn, nltk, seaborn, matplotlib
- fastapi, uvicorn
- streamlit
- lazypredict
- joblib, requests

Install all requirements:
```bash
pip install -r requirements.txt
```
Or install manually as needed.

## Usage

### 1. Data Preprocessing & Model Training
- Open `review.ipynb` in Jupyter Notebook or JupyterLab.
- Run all cells to preprocess data, train models, and save the best model and TF-IDF vectorizer:
  - `sentiment_model.pkl` (model)
  - `tfidf.pkl` (vectorizer)

### 2. Run FastAPI Backend
Start the API server:
```bash
uvicorn review_sentiments-main.api:app --reload
```
- The API will be available at `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`

#### Example API Request
Send a POST request to `/predict`:
```json
{
  "text": "This product is amazing!"
}
```
Response:
```json
{
  "label": 1,
  "label_name": "positive"
}
```

### 3. Run Streamlit Frontend
In a new terminal:
```bash
streamlit run review_sentiments-main/streamlit_app.py
```
- Open the provided local URL in your browser.
- Enter a review and click "Predict Sentiment".
