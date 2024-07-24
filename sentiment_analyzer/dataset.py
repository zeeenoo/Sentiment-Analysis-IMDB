from datasets import load_dataset
from transformers import AutoTokenizer
from .config import DATASET_NAME, MODEL_NAME, MAX_LENGTH, NUM_SAMPLES, RANDOM_SEED

def load_and_preprocess_data():
    # Load the dataset
    dataset = load_dataset(DATASET_NAME)
    
    # Limit the number of samples
    dataset = dataset.shuffle(seed=RANDOM_SEED)
    # dataset = dataset.select(range(NUM_SAMPLES))
    

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Rename label column to labels (required by Trainer)
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    # Set the format for PyTorch
    tokenized_dataset = tokenized_dataset.with_format("torch")

    return tokenized_dataset

if __name__ == "__main__":
    dataset = load_and_preprocess_data()
    print(dataset)