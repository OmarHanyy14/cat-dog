Dogs-Cats-Classification-Model

A CNN-based project to classify images as Dog or Cat, with an interactive Streamlit web application for real-time predictions.


Installation
git clone https://github.com/OmarHanyy14/cat-dog.git
cd cat-dog

python -m venv env
env\Scripts\activate  # Windows

pip install -r requirements.txt

1. Problem Definition and Data Collection

Problem: Automatically classify images into two classes: Dog or Cat.

Objective: Build an accurate and scalable image classification model and deploy it using a simple web interface.

Dataset:

Dogs vs Cats dataset from Kaggle

Dataset link: https://www.kaggle.com/datasets/tongpython/cat-and-dog

2. Data Cleaning and Analysis

Remove corrupted or invalid images (if any).

Verify class balance between dog and cat images.

Analyze image sizes and formats to ensure consistency before preprocessing.

3. Feature Engineering

Resize images to a fixed input size.

Normalize pixel values.

Apply data augmentation techniques to improve generalization:

Random flips

Random rotations

Scaling

4. Model Design

A Convolutional Neural Network (CNN) architecture is used.

The model consists of:

Convolutional layers for feature extraction

Pooling layers for dimensionality reduction

Fully connected layers for classification

Dropout layers to reduce overfitting

5. Model Training

Loss Function: Binary Crossentropy

Optimizer: Adam

Training performed on the prepared dataset.

Pre-trained weights are used to improve training speed and accuracy.

6. Model Testing and Inference

Evaluate model performance on validation/test images.

Output class probabilities for each image.

Final prediction is selected based on the highest probability score.

7. GUI Implementation and Application Running

A Streamlit web application is built to:

Upload images

Display predictions in real-time

Project division 
-------------------------------------------------
1- model design (mostafa).
2- preprocessing (omar hany).
3- training (youssef).
4- visualization (smaeh abbas - momen).
5- gui (abderahman fawzy).

Run the application using:

streamlit run Web-App.py

