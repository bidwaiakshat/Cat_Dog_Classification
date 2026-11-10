
import os
import shutil
from sklearn.model_selection import train_test_split

dataset_path = "dataset/PetImages"

image_paths = []
labels = []

class_names = ["Cat", "Dog"]

# Collect image paths and labels
for label, class_name in enumerate(class_names):
    folder = os.path.join(dataset_path, class_name)

    for filename in os.listdir(folder):
        if filename.lower().endswith(".jpg"):

            image_path = os.path.join(folder, filename)

            image_paths.append(image_path)
            labels.append(label)

print("Images collected:", len(image_paths))


# Split into training and testing sets
train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.15,
    random_state=42,
    stratify=labels
)


# Split training data into training and validation sets
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths,
    train_labels,
    test_size=0.17647,
    random_state=42,
    stratify=train_labels
)


print("Training images:", len(train_paths))
print("Validation images:", len(val_paths))
print("Testing images:", len(test_paths))

print("Total:", len(train_paths) + len(val_paths) + len(test_paths))


# Create output folders
output_path = "dataset"

splits = ["train", "validation", "test"]

for split in splits:
    for class_name in class_names:
        folder = os.path.join(output_path, split, class_name)
        os.makedirs(folder, exist_ok=True)


# Copy training images
for path, label in zip(train_paths, train_labels):
    class_name = class_names[label]

    destination = os.path.join(
        output_path, "train", class_name, os.path.basename(path)
    )

    shutil.copy2(path, destination)


# Copy validation images
for path, label in zip(val_paths, val_labels):
    class_name = class_names[label]

    destination = os.path.join(
        output_path, "validation", class_name, os.path.basename(path)
    )

    shutil.copy2(path, destination)


# Copy testing images
for path, label in zip(test_paths, test_labels):
    class_name = class_names[label]

    destination = os.path.join(
        output_path, "test", class_name, os.path.basename(path)
    )

    shutil.copy2(path, destination)


print("Images copied successfully!")