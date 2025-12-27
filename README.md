Dogs-Cats-Classification-Model 

A CNN project to classify images as Dog or Cat, with an interactive web app built using Streamlit.

Features

Accurate Classification: Dog / Cat.

Interactive Web App: Upload images and see results in real-time.

Pre-Trained Weights: For faster training and better performance.

Scalable: Works with larger datasets.

Installation
git clone https://github.com/OmarHanyy14/cat-dog.git
cd cat-dog

# Create virtual environment
python -m venv env
env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run Web-App.py

Usage
---------------------------------------------------------------

Open the Streamlit app.

Upload an image of a dog or cat.

View the classification result.

Dataset
---------------------------------------------------------------------
Trained on Dogs vs Cats dataset from Kaggle.
Kaggle Dataset Link

How It Works
-----------------------------------------------------------------------------------------
Data Preprocessing: Resize images, apply data augmentation.

Model Training: CNN trained with Binary Crossentropy and Adam optimizer.

Prediction: Uploaded images are classified with probability scores.

Web Interface: Streamlit provides a simple user interface.

Future Improvements
---------------------------------------------------------------------------------------------
Add more animal categories.

Improve UI (image cropping, drag-and-drop).

Test state-of-the-art models like EfficientNet.

Project division
---------------------------------------------------------------------------------------

Model: Mostafa Ezz Eldeen

Preprocessing: Omar Hany

Training: Yousef Ahmed

Visualization: Sameh Addas, Moamen Sabry

GUI / Web App: Abdelrahman Fawzy

## Acknowledgements

- [Kaggle Dogs vs Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)
- Tutorials and guides on CNN-based image classification.
