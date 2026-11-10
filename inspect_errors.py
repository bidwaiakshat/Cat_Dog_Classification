
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from load_dataset import test_ds
from model import model

# Load trained weights
model = tf.keras.models.load_model(
    "cats_dogs_mobilenetv2.keras"
)

from load_dataset import test_ds, test_file_paths

file_paths = test_file_paths

misclassified = []

index = 0

for images, labels in test_ds:

    predictions = model.predict(images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    actual_labels = labels.numpy()

    for i in range(len(actual_labels)):

        if actual_labels[i] != predicted_labels[i]:

            misclassified.append({
                "path": file_paths[index],
                "actual": actual_labels[i],
                "predicted": predicted_labels[i]
            })

        index += 1

print("Total incorrect predictions:", len(misclassified))

# Display first 9 incorrect predictions
plt.figure(figsize=(12, 10))

for i, item in enumerate(misclassified[:9]):

    image = tf.keras.utils.load_img(
        item["path"],
        target_size=(224, 224)
    )

    plt.subplot(3, 3, i + 1)
    plt.imshow(image)

    actual = ["Cat", "Dog"][item["actual"]]
    predicted = ["Cat", "Dog"][item["predicted"]]

    plt.title(f"Actual: {actual}\nPredicted: {predicted}")
    plt.axis("off")

plt.tight_layout()
plt.show()
