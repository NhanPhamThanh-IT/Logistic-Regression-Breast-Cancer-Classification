"""
main.py

This file contains the Gradio application for breast cancer prediction.
It loads a pre-trained model and provides a user interface for inputting features and receiving predictions.

It uses the Gradio library to create a web interface for the model.
"""

# Import necessary libraries
import gradio as gr
import pickle
import pandas as pd

# Load the pre-trained model
model = pickle.load(open("models/model.pkl", "rb"))

# Define the feature names used in the model
feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

# Function to predict the class of breast cancer based on input features
def predict(*features):
    input_data = [list(features)]
    input_df = pd.DataFrame(input_data, columns=feature_names)
    prediction = model.predict(input_df)[0]
    return "Malignant" if prediction == 0 else "Benign"

# Create the Gradio interface
with gr.Blocks(title="Breast Cancer Prediction") as demo:
    gr.Markdown("# <div align='center'>Breast Cancer Prediction</div>")
    gr.Markdown("<div align='center'>This application predicts whether a breast cancer tumor is benign or malignant based on various features.</div>")

    with gr.Row():
        with gr.Column():
            inputs_col1 = [gr.Number(label=label.title()) for label in feature_names[:10]]
        with gr.Column():
            inputs_col2 = [gr.Number(label=label.title()) for label in feature_names[10:20]]
        with gr.Column():
            inputs_col3 = [gr.Number(label=label.title()) for label in feature_names[20:]]

    predict_btn = gr.Button("Predict")
    output = gr.Textbox(label="Result")

    def on_click(*vals):
        return predict(*vals)

    predict_btn.click(fn=on_click, inputs=inputs_col1 + inputs_col2 + inputs_col3, outputs=output)

if __name__ == "__main__":
    # Launch the Gradio app
    demo.launch()
