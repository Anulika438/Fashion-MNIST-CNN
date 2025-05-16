Fashion MNIST Classification using Convolutional Neural Networks
Project Overview
This project implements a 6-layer Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset. The model is built using TensorFlow and Keras, and achieves over 90% accuracy in classifying clothing items into 10 different categories.
Dataset
The Fashion MNIST dataset consists of 70,000 grayscale images of fashion products from 10 categories, with 60,000 training images and 10,000 test images. Each image is 28x28 pixels in grayscale.
The dataset categories are:
| Label | Description |
|-------|-------------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |
The implemented CNN has 6 layers:

Convolutional Layer (32 filters, 3x3 kernel)
Max Pooling Layer (2x2)
Convolutional Layer (64 filters, 3x3 kernel)
Max Pooling Layer (2x2)
Fully Connected Layer (128 neurons with dropout)
Output Layer (10 neurons with softmax activation)

Results
The model achieves approximately 92% accuracy on the test set, demonstrating effective learning of clothing item features. Specific predictions were made on two test images:

Image 42: True class: Dress, Predicted class: Shirt (75.57% confidence)
Image 123: True class: Ankle boot, Predicted class: Ankle boot (100.00% confidence)

Files Description

fashion_mnist_cnn.py: Main Python script containing the CNN implementation
fashion_mnist_cnn_model.h5: Saved trained model
fashion_mnist_examples.png: Visualization of sample images from the dataset
training_history.png: Plots of training and validation metrics over epochs
model_predictions.png: Visualization of model predictions on random test images
confusion_matrix.png: Confusion matrix showing classification performance by category
specific_prediction_42.png: Detailed prediction visualization for test image #42
specific_prediction_123.png: Detailed prediction visualization for test image #123

Requirements

Python 3.10 or higher
TensorFlow 2.15 or higher
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn

How to Run
To run the project, execute the main script:
python fashion_mnist_cnn.py
The script will:

Load and preprocess the Fashion MNIST dataset
Build and train the CNN model
Evaluate the model's performance
Generate visualizations of results
Make predictions on specific test images


Anulika Blessing Ekwuno
15th May, 2025