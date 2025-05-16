import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras
layers = tf.keras.layers
models = tf.keras.models
fashion_mnist = tf.keras.datasets.fashion_mnist
to_categorical = tf.keras.utils.to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Load the Fashion MNIST dataset
print("Loading the Fashion MNIST dataset...")
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Get dataset information
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# 2. Data Preprocessing
# Normalize pixel values to be between 0 and 1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape the data to fit the model (28x28x1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# One-hot encode the target variable
y_train_categorical = to_categorical(y_train, num_classes=10)
y_test_categorical = to_categorical(y_test, num_classes=10)

# Define class names for better visualization
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 3. Visualize some examples from the dataset
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
plt.suptitle('Fashion MNIST Examples', fontsize=16)
plt.savefig('fashion_mnist_examples.png', dpi=300, bbox_inches='tight')
plt.close()


# 4. Build a 6-layer CNN model
def create_model():
    model = models.Sequential([
        # Layer 1: Convolutional Layer
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),

        # Layer 2: Max Pooling Layer
        layers.MaxPooling2D((2, 2)),

        # Layer 3: Convolutional Layer
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),

        # Layer 4: Max Pooling Layer
        layers.MaxPooling2D((2, 2)),

        # Layer 5: Fully Connected Layer
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Add dropout to prevent overfitting

        # Layer 6: Output Layer
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Create and summarize the model
model = create_model()
model.summary()

# 5. Train the model
print("\nTraining the model...")
history = model.fit(
    X_train, y_train_categorical,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# 6. Evaluate the model
print("\nEvaluating the model...")
test_loss, test_acc = model.evaluate(X_test, y_test_categorical, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

# 7. Visualize training history
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. Make predictions for some test images
num_images_to_predict = 10
random_indices = np.random.choice(len(X_test), num_images_to_predict, replace=False)

# Get the images and their true labels
images_to_predict = X_test[random_indices]
true_labels = y_test[random_indices]

# Make predictions
predictions = model.predict(images_to_predict)
predicted_labels = np.argmax(predictions, axis=1)

# Display the images and predictions
plt.figure(figsize=(15, 8))
for i in range(num_images_to_predict):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images_to_predict[i].reshape(28, 28), cmap=plt.cm.binary)
    color = 'green' if predicted_labels[i] == true_labels[i] else 'red'
    plt.title(f"Pred: {class_names[predicted_labels[i]]}\nTrue: {class_names[true_labels[i]]}", color=color)
    plt.axis('off')
plt.suptitle('Model Predictions', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig('model_predictions.png', dpi=300, bbox_inches='tight')
plt.close()

# 9. Generate confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Create and plot confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 10. Generate classification report
print("\nClassification Report:")
report = classification_report(y_test, y_pred_classes, target_names=class_names)
print(report)

# Save the model
model.save('fashion_mnist_cnn_model.h5')
print("Model saved as 'fashion_mnist_cnn_model.h5'")


# 11. Class for detailed prediction on specific images
class FashionPredictor:
    def __init__(self, model_path='fashion_mnist_cnn_model.h5'):
        self.model = keras.models.load_model(model_path)
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def predict_specific_image(self, image_idx, test_images, test_labels):
        """Make a prediction for a specific image from the test set."""
        if image_idx >= len(test_images):
            raise ValueError(f"Image index {image_idx} is out of bounds.")

        # Get the image and reshape it
        image = test_images[image_idx].reshape(1, 28, 28, 1)
        true_label = test_labels[image_idx]

        # Make prediction
        prediction = self.model.predict(image, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class] * 100

        # Display image and prediction
        plt.figure(figsize=(6, 6))
        plt.imshow(image.reshape(28, 28), cmap=plt.cm.binary)

        # Set title color based on correctness
        color = 'green' if predicted_class == true_label else 'red'

        plt.title(f"Prediction: {self.class_names[predicted_class]} ({confidence:.2f}%)\n"
                  f"True: {self.class_names[true_label]}", color=color)

        plt.axis('off')
        plt.tight_layout()

        # Save and display the result
        plt.savefig(f'specific_prediction_{image_idx}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Return the results
        return {
            'image_idx': image_idx,
            'true_class': self.class_names[true_label],
            'predicted_class': self.class_names[predicted_class],
            'confidence': confidence,
            'correct': predicted_class == true_label
        }


# Create predictor instance
predictor = FashionPredictor('fashion_mnist_cnn_model.h5')

# Predict two specific images as required by the assignment
print("\nMaking predictions for two specific images:")
image_indices = [42, 123]  # Choose two images for detailed prediction
results = []

for idx in image_indices:
    result = predictor.predict_specific_image(idx, X_test, y_test)
    results.append(result)
    print(f"Image {idx}:")
    print(f"  True class: {result['true_class']}")
    print(f"  Predicted class: {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']:.2f}%")
    print(f"  Correct: {result['correct']}")
    print()


# 12. Additional visualization: Activation Maps
def visualize_activations(image_idx):
    """Visualize the activation maps for a specific image."""
    if image_idx >= len(X_test):
        raise ValueError(f"Image index {image_idx} is out of bounds.")

    # Get the image
    image = X_test[image_idx:image_idx + 1]

    # Create a model that returns the activations of the first convolutional layer
    activation_model = models.Model(
        inputs=model.input,
        outputs=[
            model.layers[0].output,  # First conv layer
            model.layers[2].output  # Second conv layer
        ]
    )

    # Get activations
    activations = activation_model.predict(image)

    # Show input image
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(image[0].reshape(28, 28), cmap='gray')
    plt.title(f"Input Image\n({class_names[y_test[image_idx]]})")
    plt.axis('off')

    # Show activations from first conv layer
    plt.subplot(1, 3, 2)
    plt.imshow(np.mean(activations[0][0], axis=-1), cmap='viridis')
    plt.title("Activation Map - Conv Layer 1")
    plt.axis('off')

    # Show activations from second conv layer
    plt.subplot(1, 3, 3)
    plt.imshow(np.mean(activations[1][0], axis=-1), cmap='viridis')
    plt.title("Activation Map - Conv Layer 2")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'activation_maps_{image_idx}.png', dpi=300, bbox_inches='tight')
    plt.close()


# Visualize activations for the two specific images
#for idx in image_indices:
#    visualize_activations(idx)

print("All visualizations have been saved as PNG files.")
print("Assignment completed successfully!")