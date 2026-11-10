
import tensorflow as tf
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

from load_dataset import test_ds


# Load trained model
model = tf.keras.models.load_model(
    "cats_dogs_mobilenetv2.keras"
)


# Evaluate model using Keras
test_loss, test_accuracy = model.evaluate(test_ds)

print("\nMODEL EVALUATION")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy * 100, "%")


# Generate predictions for classification metrics
y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(images, verbose=0)

    predicted_labels = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)


# Calculate accuracy
print("\nACCURACY SCORE")
print("Accuracy:", accuracy_score(y_true, y_pred))


# Confusion matrix
print("\nCONFUSION MATRIX")

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)


# Classification report
print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Cat", "Dog"]
    )
)
