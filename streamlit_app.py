import streamlit as st
import requests

st.title('Sentiment Analysis App')

st.write('Enter a product review below and get the predicted sentiment!')

review_text = st.text_area('Review Text', '')

if st.button('Predict Sentiment'):
    if not review_text.strip():
        st.warning('Please enter some text.')
    else:
        # Send POST request to FastAPI
        try:
            response = requests.post(
                'http://localhost:8000/predict',
                json={'text': review_text}
            )
            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted Sentiment: {result['label_name'].capitalize()} (label: {result['label']})")
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Could not connect to API: {e}") 