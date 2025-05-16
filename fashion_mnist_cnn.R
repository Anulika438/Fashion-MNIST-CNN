# Fashion MNIST Classification using CNN in R
# Load required libraries
library(keras)
library(tensorflow)
library(tidyverse)
library(caret)
library(ggplot2)

# Set random seed for reproducibility
set.seed(42)
tensorflow::tf$random$set_seed(42)

# 1. Load the Fashion MNIST dataset
cat("Loading Fashion MNIST dataset...\n")
fashion_mnist <- dataset_fashion_mnist()

# Split into training and test sets
c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

# Get dataset information
cat(sprintf("Training data shape: %s x %s\n", dim(train_images)[1], paste(dim(train_images)[2:3], collapse=" x ")))
cat(sprintf("Test data shape: %s x %s\n", dim(test_images)[1], paste(dim(test_images)[2:3], collapse=" x ")))

# Define class names for better visualization
class_names <- c('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

# 2. Data Preprocessing
# Normalize pixel values to be between 0 and 1
train_images <- train_images / 255
test_images <- test_images / 255

# Reshape the data to fit the model (add channel dimension)
train_images <- array_reshape(train_images, c(dim(train_images), 1))
test_images <- array_reshape(test_images, c(dim(test_images), 1))

# Convert labels to categorical
train_labels_categorical <- to_categorical(train_labels, 10)
test_labels_categorical <- to_categorical(test_labels, 10)

# 3. Build a 6-layer CNN model
model <- keras_model_sequential() %>%
  # Layer 1: Convolutional Layer
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu',
                input_shape = c(28, 28, 1), padding = 'same') %>%
  
  # Layer 2: Max Pooling Layer
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  # Layer 3: Convolutional Layer
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu',
                padding = 'same') %>%
  
  # Layer 4: Max Pooling Layer
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  # Layer 5: Fully Connected Layer
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  
  # Layer 6: Output Layer
  layer_dense(units = 10, activation = 'softmax')

# Compile the model
model %>% compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

# Display model summary
summary(model)

# 4. Train the model
cat("\nTraining the model...\n")
history <- model %>% fit(
  x = train_images,
  y = train_labels_categorical,
  epochs = 10,
  batch_size = 64,
  validation_split = 0.2,
  verbose = 1
)

# 5. Evaluate the model
cat("\nEvaluating the model...\n")
scores <- model %>% evaluate(test_images, test_labels_categorical, verbose = 0)
cat("Test loss:", scores[[1]], "\n")
cat("Test accuracy:", scores[[2]], "\n")

# 6. Plot training history
png("training_history_r.png", width = 1200, height = 500, res = 100)
par(mfrow = c(1, 2))
plot(history$metrics$accuracy, type = "l", col = "blue", 
     xlab = "Epoch", ylab = "Accuracy", main = "Model Accuracy")
lines(history$metrics$val_accuracy, col = "red")
legend("bottomright", legend = c("Training", "Validation"), 
       col = c("blue", "red"), lty = 1)

plot(history$metrics$loss, type = "l", col = "blue", 
     xlab = "Epoch", ylab = "Loss", main = "Model Loss")
lines(history$metrics$val_loss, col = "red")
legend("topright", legend = c("Training", "Validation"), 
       col = c("blue", "red"), lty = 1)
dev.off()

# 7. Make predictions for specific test images
cat("\nMaking predictions for specific images...\n")

# Function to visualize predictions for a specific image
predict_specific_image <- function(image_idx) {
  # Get the image and reshape it
  image <- array_reshape(test_images[image_idx,,,,drop=FALSE], c(1, 28, 28, 1))
  true_label <- test_labels[image_idx] + 1  # Add 1 for R indexing
  
  # Make prediction
  prediction <- model %>% predict(image)
  predicted_class <- which.max(prediction[1,])
  confidence <- prediction[1, predicted_class] * 100
  
  # Create visualization
  png(paste0("specific_prediction_r_", image_idx, ".png"), width = 600, height = 600, res = 100)
  par(mar = c(1, 1, 4, 1))
  image_to_plot <- array_reshape(image, c(28, 28))
  image(t(apply(image_to_plot, 2, rev)), axes = FALSE, col = gray.colors(100), 
        main = paste0("Prediction: ", class_names[predicted_class], 
                      " (", round(confidence, 2), "%)\n",
                      "True: ", class_names[true_label]))
  dev.off()
  
  # Return results
  result <- list(
    image_idx = image_idx,
    true_class = class_names[true_label],
    predicted_class = class_names[predicted_class],
    confidence = confidence,
    correct = predicted_class == true_label
  )
  
  return(result)
}

# Predict two specific images
image_indices <- c(42, 123)
results <- list()

for (idx in image_indices) {
  result <- predict_specific_image(idx)
  results[[length(results) + 1]] <- result
  cat(paste0("Image ", idx, ":\n"))
  cat(paste0("  True class: ", result$true_class, "\n"))
  cat(paste0("  Predicted class: ", result$predicted_class, "\n"))
  cat(paste0("  Confidence: ", round(result$confidence, 2), "%\n"))
  cat(paste0("  Correct: ", result$correct, "\n\n"))
}

# 8. Create confusion matrix
cat("\nGenerating confusion matrix...\n")
y_pred <- model %>% predict(test_images)
y_pred_classes <- max.col(y_pred) - 1  # Convert to 0-based indexing
conf_matrix <- confusionMatrix(factor(y_pred_classes), factor(test_labels))

# Save confusion matrix visualization
library(lattice)
png("confusion_matrix_r.png", width = 1200, height = 1000, res = 100)
levelplot(conf_matrix$table, 
          main = "Confusion Matrix",
          xlab = "Predicted", ylab = "Actual",
          col.regions = colorRampPalette(c("white", "blue"))(100))
dev.off()

print(conf_matrix)

# Save the model
model %>% save_model_hdf5("fashion_mnist_cnn_model_r.h5")
cat("Model saved as 'fashion_mnist_cnn_model_r.h5'\n")

cat("\nR implementation completed successfully!\n")