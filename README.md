# Sentiment Analysis with Hugging Face and Streamlit

## Overview

This project demonstrates an end-to-end sentiment analysis application using Hugging Face's transformers library and Streamlit. It allows users to train a sentiment analysis model on the IMDB dataset, evaluate its performance, and make predictions on new text inputs through an interactive web interface.

## Features

- **Data Loading**: Utilizes Hugging Face's datasets library to load the IMDB dataset without local downloads.
- **Model Training**: Fine-tunes a pre-trained DistilBERT model on the IMDB dataset for sentiment analysis.
- **Model Evaluation**: Provides performance metrics including accuracy, precision, recall, and F1 score.
- **Sentiment Prediction**: Allows users to input text and receive sentiment predictions with confidence scores.
- **Interactive UI**: Built with Streamlit for an intuitive and responsive user experience.
- **Visualizations**: Incorporates Plotly for creating interactive charts to display model performance and prediction confidence.

## Prerequisites

- Python 3.7+
- pip (Python package installer)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/sentiment-analysis-streamlit.git
   cd sentiment-analysis-streamlit
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the Streamlit app:

```
streamlit run app.py
```

This will start a local server and open the app in your default web browser.

### Navigation

The app consists of four main sections:

1. **Home**: Provides an introduction to the sentiment analysis application.
2. **Train Model**: Allows users to initiate the model training process.
3. **Evaluate Model**: Displays the performance metrics of the trained model using interactive bar charts.
4. **Predict**: Enables users to input text and receive sentiment predictions with confidence scores visualized using a gauge chart.

## Project Structure

```
project_root/
├── README.md
├── requirements.txt
├── app.py
└── sentiment_analyzer/
    ├── __init__.py
    ├── config.py
    ├── dataset.py
    ├── model.py
    ├── train.py
    ├── evaluate.py
    └── predict.py
```

- `app.py`: The main Streamlit application file.
- `sentiment_analyzer/`: A package containing the core functionality for sentiment analysis.
  - `config.py`: Configuration settings for the project.
  - `dataset.py`: Handles dataset loading and preprocessing.
  - `model.py`: Defines the sentiment analysis model.
  - `train.py`: Contains the training logic.
  - `evaluate.py`: Implements model evaluation.
  - `predict.py`: Provides functionality for making predictions.

## Customization

You can customize various aspects of the project by modifying the `config.py` file:

- `DATASET_NAME`: Change the dataset used for training.
- `NUM_SAMPLES`: Adjust the number of samples used from the dataset.
- `MODEL_NAME`: Use a different pre-trained model.
- `MAX_LENGTH`: Modify the maximum sequence length for tokenization.
- `BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`: Tune the training hyperparameters.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Hugging Face for their transformers and datasets libraries.
- Streamlit for the amazing web app framework.
- The creators and contributors of the IMDB dataset.

## Contact

For any questions or feedback, please open an issue in the GitHub repository or contact [ME Here](mailto:a_marhoum@estin.dz).
