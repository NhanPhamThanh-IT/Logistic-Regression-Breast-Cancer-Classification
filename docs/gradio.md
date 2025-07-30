# <div align="center">Gradio: Interactive Machine Learning Demos Made Easy</div>

<div align="justify">

## Table of Contents

1. [What is Gradio?](#what-is-gradio)
2. [Why Use Gradio?](#why-use-gradio)
3. [Installing Gradio](#installing-gradio)
4. [Basic Concepts](#basic-concepts)
5. [Building a Simple Gradio Interface](#building-a-simple-gradio-interface)
6. [Integrating Gradio with Handwritten Digit Classification](#integrating-gradio-with-handwritten-digit-classification)
7. [Advanced Features](#advanced-features)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Further Resources](#further-resources)

---

## What is Gradio?

**Gradio** is an open-source Python library that allows you to quickly create user-friendly web interfaces for your machine learning models, data science workflows, or any Python function. With just a few lines of code, you can turn your model into an interactive demo that runs in your browser.

- **Website:** [https://gradio.app/](https://gradio.app/)
- **GitHub:** [https://github.com/gradio-app/gradio](https://github.com/gradio-app/gradio)

---

## Why Use Gradio?

- **Instant Demos:** Share your model with non-technical users or stakeholders.
- **Rapid Prototyping:** Test your model’s behavior with real inputs.
- **Collaboration:** Share a link to your interface with anyone, anywhere.
- **No Frontend Skills Needed:** Gradio handles all the web UI for you.

---

## Installing Gradio

You can install Gradio using pip:

```bash
pip install gradio
```

If you are using a virtual environment (recommended), make sure it is activated before installing.

---

## Basic Concepts

### 1. **Interface**

The core of Gradio is the `Interface` class, which connects:

- **A Python function** (your model or processing logic)
- **Input components** (e.g., image upload, text box)
- **Output components** (e.g., label, image, text)

### 2. **Launching**

Calling `.launch()` on an Interface object starts a local web server and opens the UI in your browser.

### 3. **Components**

Gradio provides many input/output components:

- `gr.inputs.Image`, `gr.inputs.Textbox`, `gr.outputs.Label`, etc.
- In Gradio 3.x+, use `gr.Image`, `gr.Textbox`, `gr.Label`, etc.

---

## Building a Simple Gradio Interface

Here’s a minimal example for a digit classifier:

```python
import gradio as gr
import numpy as np
from tensorflow import keras

# Load your trained model
model = keras.models.load_model("models/model.keras")

def predict_digit(image):
    # Preprocess the image as your model expects
    image = image.reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict(image)
    return int(np.argmax(prediction))

# Create the interface
iface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(shape=(28, 28), image_mode='L', invert_colors=True, source="canvas"),
    outputs=gr.Label(num_top_classes=3),
    live=False,
    title="Handwritten Digit Classifier",
    description="Draw a digit (0-9) and let the model predict it!"
)

iface.launch()
```

**Explanation:**

- `fn`: The function to call when the user submits input.
- `inputs`: The input component (here, a 28x28 grayscale canvas).
- `outputs`: The output component (top-3 predicted labels).
- `live`: If True, updates output as you draw.
- `title`/`description`: Shown at the top of the UI.

---

## Integrating Gradio with Handwritten Digit Classification

### 1. **Preprocessing**

Your model expects images in a certain format (e.g., 28x28 grayscale, normalized). Make sure your Gradio input matches this.

### 2. **Example Usage**

Suppose your model is trained on the MNIST dataset. Here’s how you might set up the interface:

```python
def predict_digit(image):
    import numpy as np
    # Convert to numpy array, resize, normalize, etc.
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict(image)
    return {str(i): float(prediction[0][i]) for i in range(10)}

iface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(shape=(28, 28), image_mode='L', invert_colors=True, source="canvas"),
    outputs=gr.Label(num_top_classes=3),
    examples=[
        ["app/assets/mnist.jpg"]
    ],
    title="Handwritten Digit Classifier",
    description="Draw a digit or upload an image. The model will predict the digit."
)
iface.launch()
```

### 3. **Adding Examples**

You can provide example images for users to try out.

---

## Advanced Features

### 1. **Multiple Inputs/Outputs**

Gradio supports multiple inputs and outputs. For example, you can accept both an image and a text label, or return both a label and a confidence score.

### 2. **Custom Layouts**

Use `gr.Blocks` for more complex UIs (multiple components, custom layouts).

### 3. **Sharing Public Links**

Set `share=True` in `.launch()` to get a public link (useful for sharing with others).

```python
iface.launch(share=True)
```

### 4. **Integrating with Notebooks**

Gradio works seamlessly in Jupyter and Colab notebooks.

---

## Best Practices

- **Preprocess Inputs:** Always preprocess user input to match your model’s training data.
- **Handle Errors:** Add try/except blocks in your prediction function to handle unexpected input.
- **Limit Input Size:** For images, specify the expected shape and mode.
- **Document Your Interface:** Use `title`, `description`, and `article` to explain your model and its limitations.
- **Security:** Never expose sensitive models or data via public links unless you intend to.

---

## Troubleshooting

- **Interface Not Launching:** Check for port conflicts or firewall issues.
- **Model Not Loading:** Ensure the model path is correct and dependencies are installed.
- **Input Shape Errors:** Double-check that your preprocessing matches the model’s expected input.
- **Gradio Version Issues:** Some APIs changed between Gradio 2.x and 3.x+. Check the [migration guide](https://gradio.app/docs/migration_guide/).

---

## Further Resources

- [Gradio Documentation](https://gradio.app/docs/)
- [Gradio Tutorials](https://gradio.app/get_started/)
- [Gradio GitHub](https://github.com/gradio-app/gradio)
- [MNIST Dataset Info](mnist-dataset.md)
- [TensorFlow Model Guide](tensorflow.md)

---

## Example: Full Integration in Your Project

Suppose your main app is in `app/main.py`. Here’s how you might structure it:

```python
import gradio as gr
import numpy as np
from tensorflow import keras

model = keras.models.load_model("models/model.keras")

def predict_digit(image):
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict(image)
    return {str(i): float(prediction[0][i]) for i in range(10)}

iface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(shape=(28, 28), image_mode='L', invert_colors=True, source="canvas"),
    outputs=gr.Label(num_top_classes=3),
    title="Handwritten Digit Classifier",
    description="Draw a digit or upload an image. The model will predict the digit.",
    examples=[["app/assets/mnist.jpg"]]
)

if __name__ == "__main__":
    iface.launch()
```

---

# <div align="center">Summary</div>

Gradio is a powerful tool for making your machine learning models interactive and accessible. With just a few lines of code, you can create demos, collect feedback, and share your work with the world.

</div>

---

<div align="center">

**Happy coding and experimenting with Gradio!**

</div>
