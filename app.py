import streamlit as st
import plotly.graph_objects as go
from sentiment_analyzer.predict import predict
from sentiment_analyzer.train import train
from sentiment_analyzer.evaluate import evaluate
import time

def main():
    st.set_page_config(page_title="Sentiment Analysis App", page_icon="😊")
    st.title("Sentiment Analysis with Hugging Face and Streamlit")

    menu = ["Home", "Train Model", "Evaluate Model", "Predict"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Welcome to the Sentiment Analysis App!")
        st.write("This app demonstrates sentiment analysis using Hugging Face models and datasets.")
        st.write("Use the menu on the left to navigate through different sections of the app.")

    elif choice == "Train Model":
        st.subheader("Train the Sentiment Analysis Model")
        if st.button("Start Training"):
            with st.spinner("Training in progress..."):
                train()
            st.success("Training completed!")

    elif choice == "Evaluate Model":
        st.subheader("Evaluate the Sentiment Analysis Model")
        if st.button("Start Evaluation"):
            with st.spinner("Evaluation in progress..."):
                results = evaluate()
            
            st.write("Evaluation Results:")
            fig = go.Figure(data=[go.Bar(x=list(results.keys()), y=list(results.values()))])
            fig.update_layout(title="Model Performance Metrics", xaxis_title="Metrics", yaxis_title="Score")
            st.plotly_chart(fig)

    elif choice == "Predict":
        st.subheader("Predict Sentiment")
        user_input = st.text_area("Enter your text here:")
        if st.button("Predict"):
            if user_input:
                with st.spinner("Analyzing sentiment..."):
                    sentiment, confidence = predict(user_input)
                
                st.write(f"Sentiment: {sentiment}")
                st.write(f"Confidence: {confidence:.4f}")

                # Create a gauge chart for confidence
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = confidence,
                    title = {'text': "Confidence"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {'axis': {'range': [0, 1]},
                             'bar': {'color': "darkblue"},
                             'steps' : [
                                 {'range': [0, 0.33], 'color': "lightgray"},
                                 {'range': [0.33, 0.66], 'color': "gray"},
                                 {'range': [0.66, 1], 'color': "darkgray"}],
                             'threshold': {
                                 'line': {'color': "red", 'width': 4},
                                 'thickness': 0.75,
                                 'value': 0.9}}))
                st.plotly_chart(fig)
            else:
                st.warning("Please enter some text for sentiment analysis.")

if __name__ == "__main__":
    main()