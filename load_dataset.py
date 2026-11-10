import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Dataset directories
train_dir = "dataset/train"
val_dir = "dataset/validation"
test_dir = "dataset/test"

# Load training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode="int"
)

# Load validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode="int"
)

# Load testing dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode="int",
    shuffle=False
)
# Save file paths before preprocessing
test_file_paths = test_ds.file_paths
print("Classes:", train_ds.class_names)

for images, labels in train_ds.take(1):
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("First 10 labels:", labels[:10].numpy())

for images, labels in train_ds.take(1):

    print("Minimum pixel value:", tf.reduce_min(images).numpy())
    print("Maximum pixel value:", tf.reduce_max(images).numpy())
    print("Data type:", images.dtype)


    # Preprocess images for MobileNetV2
train_ds = train_ds.map(
    lambda images, labels: (preprocess_input(images), labels)
)

val_ds = val_ds.map(
    lambda images, labels: (preprocess_input(images), labels)
)

test_ds = test_ds.map(
    lambda images, labels: (preprocess_input(images), labels)
)

for images, labels in train_ds.take(1):
    print("Minimum:", tf.reduce_min(images).numpy())
    print("Maximum:", tf.reduce_max(images).numpy())
    print("Shape:", images.shape)

