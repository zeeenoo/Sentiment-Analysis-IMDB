from transformers import TrainingArguments, Trainer
from .dataset import load_and_preprocess_data
from .model import load_model
from .config import MODEL_NAME, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS

def train():
    # Load and preprocess the dataset
    dataset = load_and_preprocess_data()
    
    # Load the model
    model = load_model(MODEL_NAME)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=LEARNING_RATE,
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"].select(range(1000)),
        eval_dataset=dataset["test"].select(range(500)),
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model("./sentiment_model")

if __name__ == "__main__":
    train()