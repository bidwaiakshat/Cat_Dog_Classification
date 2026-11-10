import tensorflow as tf
from load_dataset import test_ds
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

finetuned_model = tf.keras.models.load_model(
    "cats_dogs_finetuned.keras"
)

test_loss, test_accuracy = finetuned_model.evaluate(test_ds)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

y_true = np.concatenate([labels.numpy() for images, labels in test_ds])

y_prob = finetuned_model.predict(test_ds)

y_pred = np.argmax(y_prob, axis=1)

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Cat", "Dog"]
))