from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from .config import MODEL_NAME

def predict(text):
    # Load the fine-tuned model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Get the predicted class (0 for negative, 1 for positive)
    predicted_class = prediction.argmax().item()
    confidence = prediction[0][predicted_class].item()

    return "Positive" if predicted_class == 1 else "Negative", confidence

if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "This movie was fantastic! I really enjoyed every moment of it.",
        "I've never been so bored in my life. Awful film.",
        "It was okay, I guess. Nothing special but not terrible either."
    ]

    for text in sample_texts:
        sentiment, confidence = predict(text)
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.4f}")
        print()