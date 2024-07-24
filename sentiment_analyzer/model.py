from transformers import AutoModelForSequenceClassification

def load_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model