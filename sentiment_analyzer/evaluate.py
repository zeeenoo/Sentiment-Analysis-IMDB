from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from .config import DATASET_NAME, NUM_SAMPLES, RANDOM_SEED, MAX_LENGTH, MODEL_NAME

def evaluate():
    # Load the fine-tuned model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load the test dataset
    dataset = load_dataset(DATASET_NAME, split="test")
    dataset = dataset.shuffle(seed=RANDOM_SEED)
    dataset = dataset.select(range(NUM_SAMPLES))

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Make predictions
    model.eval()
    predictions = []
    for batch in tokenized_dataset:
        with torch.no_grad():
            outputs = model(**{k: torch.tensor([v]) for k, v in batch.items() if k in ["input_ids", "attention_mask"]})
            predictions.append(outputs.logits.argmax().item())

    # Calculate metrics
    labels = dataset["label"]
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

if __name__ == "__main__":
    results = evaluate()
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")