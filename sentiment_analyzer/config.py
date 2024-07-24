# Dataset configuration
DATASET_NAME = "imdb"
NUM_SAMPLES = 10000  # Limit the number of samples to speed up processing

# Model configuration
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_LENGTH = 128

# Training configuration
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3

# Misc
RANDOM_SEED = 42

# Streamlit configuration
MODEL_PATH = "./sentiment_model"