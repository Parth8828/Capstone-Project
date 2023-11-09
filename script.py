import requests
from transformers import BertTokenizer
import numpy as np
import re
import nltk

sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
nltk.download('stopwords')

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define a function for text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    stop_words = set(nltk.corpus.stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    text = " ".join(words)
    return text

# Function to get sentiment prediction from TensorFlow Serving
def get_sentiment_prediction(input_text):
    # Preprocess the input text
    preprocessed_input = preprocess_text(input_text)

    # Tokenize and prepare the input data
    input_encodings = tokenizer(preprocessed_input, truncation=True, padding=True, max_length=64, return_tensors='tf')
    input_ids = input_encodings["input_ids"]

    # Define the URL for the TensorFlow Serving API
    api_url = "http://localhost:8601/v1/models/sentiment_model:predict"  # Adjust as needed

    # Prepare the request payload with the input_ids
    # Prepare the request payload with all expected keys
    payload = {
    "signature_name": "serving_default",
    "instances": [{
        "input_ids": input_ids[0].numpy().tolist(),
        "attention_mask": [1] * len(input_ids[0]),  # Assuming all tokens are attended to
        "token_type_ids": [0] * len(input_ids[0])  # Assuming single-segment input
    }]
    }

    # Send a POST request to the API
    response = requests.post(api_url, json=payload)

    if response.status_code == 200:
        predictions = response.json()["predictions"][0]
        sentiment_class = sentiment_labels[np.argmax(predictions)]
        return sentiment_class
    else:
        print(f"Error: Failed to get sentiment prediction. Status Code: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    print("BERT Sentiment Analysis with TensorFlow Serving")
    print("Enter 'exit' to quit.")

    while True:
        user_input = input("Enter a review or chat: ")
        if user_input.lower() == 'exit':
            break

        sentiment_prediction = get_sentiment_prediction(user_input)
        print(f"Predicted Sentiment: {sentiment_prediction}")
