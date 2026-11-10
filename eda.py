
import os
import io
import zipfile
import warnings
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image


# ==================================================
# CONFIGURATION
# ==================================================

ZIP_PATH = "cats_and_dogs.zip"
CLEANED_ZIP_PATH = "cats_and_dogs_cleaned.zip"

DATASET_PATH = "dataset"

dataset_folders = [
    "train/Cat",
    "train/Dog",
    "validation/Cat",
    "validation/Dog",
    "test/Cat",
    "test/Dog"
]

IMAGE_SIZE = (224, 224)

corrupted_zip_images = []


# ==================================================
# 1. DATASET DISTRIBUTION
# ==================================================

def analyze_dataset():

    print("\nDATASET DISTRIBUTION")

    cat_count = 0
    dog_count = 0
    formats = Counter()

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:

        for file in zip_ref.namelist():

            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            if "/Cat/" in file:
                cat_count += 1

            elif "/Dog/" in file:
                dog_count += 1

            extension = os.path.splitext(file)[1].lower()
            formats[extension] += 1

    print("Cat:", cat_count)
    print("Dog:", dog_count)
    print("Total images:", cat_count + dog_count)

    print("\nIMAGE FORMATS")

    for extension, count in formats.items():
        print(f"{extension}: {count}")


# ==================================================
# 2. IMAGE DIMENSIONS AND CORRUPTION CHECK
# ==================================================

def analyze_dimensions():

    print("\nIMAGE DIMENSIONS")

    dimensions = Counter()
    valid_images = 0

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:

        for file in zip_ref.namelist():

            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            try:
                image_data = zip_ref.read(file)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    with Image.open(io.BytesIO(image_data)) as image:
                        image.verify()

                    with Image.open(io.BytesIO(image_data)) as image:
                        image.load()
                        dimensions[image.size] += 1

                valid_images += 1

            except Exception:
                corrupted_zip_images.append(file)

    print("Valid images:", valid_images)
    print("Different dimensions:", len(dimensions))

    print("\nMost common dimensions:")

    for size, count in dimensions.most_common(10):
        print(size, ":", count)

    print("\nCorrupted images:", len(corrupted_zip_images))

    for file in corrupted_zip_images:
        print(file)


# ==================================================
# 3. CREATE CLEANED ZIP
# ==================================================

def remove_corrupted_from_zip():

    print("\nCREATING CLEANED ZIP")

    corrupted_set = set(corrupted_zip_images)

    if not corrupted_set:
        print("No corrupted images found.")
        return

    removed = 0

    with zipfile.ZipFile(ZIP_PATH, "r") as original_zip:

        with zipfile.ZipFile(
            CLEANED_ZIP_PATH, "w", zipfile.ZIP_DEFLATED
        ) as cleaned_zip:

            for item in original_zip.infolist():

                if item.filename in corrupted_set:
                    removed += 1
                    print("Excluded:", item.filename)
                    continue

                cleaned_zip.writestr(
                    item,
                    original_zip.read(item.filename)
                )

    print("Corrupted images excluded:", removed)
    print("Cleaned ZIP created:", CLEANED_ZIP_PATH)
    print("Original ZIP preserved.")


# ==================================================
# 4. SAMPLE IMAGE PREPROCESSING
# ==================================================

def demonstrate_preprocessing():

    print("\nSAMPLE IMAGE PREPROCESSING")

    corrupted_set = set(corrupted_zip_images)

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:

        sample_file = None

        for file in zip_ref.namelist():

            if "/Cat/" not in file:
                continue

            if not file.lower().endswith((".jpg", ".jpeg")):
                continue

            if file in corrupted_set:
                continue

            sample_file = file
            break

        if sample_file is None:
            print("No valid sample image found.")
            return

        image_data = zip_ref.read(sample_file)

        with Image.open(io.BytesIO(image_data)) as image:

            print("Sample image:", sample_file)
            print("Original size:", image.size)
            print("Original mode:", image.mode)

            image = image.convert("RGB")
            image = image.resize(IMAGE_SIZE)

            image_array = np.array(image)

            print("New size:", image.size)
            print("Array shape:", image_array.shape)
            print("Data type:", image_array.dtype)

            plt.imshow(image_array)
            plt.title("Sample Image After Preprocessing")
            plt.axis("off")
            plt.show()


# ==================================================
# 5. CHECK AND CLEAN PREPARED DATASET
# ==================================================

def clean_prepared_dataset():

    print("\nCHECKING AND CLEANING PREPARED DATASET")

    total_checked = 0
    already_rgb = 0
    converted = 0
    deleted = 0
    failed = 0

    for folder in dataset_folders:

        folder_path = os.path.join(DATASET_PATH, folder)

        if not os.path.isdir(folder_path):
            print("Folder not found:", folder_path)
            continue

        print("\nChecking:", folder_path)

        for filename in os.listdir(folder_path):

            if not filename.lower().endswith((".jpg", ".jpeg")):
                continue

            image_path = os.path.join(folder_path, filename)

            if not os.path.isfile(image_path):
                continue

            total_checked += 1

            try:

                # First, check whether TensorFlow can decode the image
                image_data = tf.io.read_file(image_path)

                decoded_image = tf.io.decode_jpeg(
                    image_data,
                    channels=3,
                    try_recover_truncated=False
                )

                decoded_image.numpy()

                # Next, check the image's actual colour mode
                with Image.open(image_path) as image:

                    image.load()

                    if image.mode != "RGB":

                        rgb_image = image.convert("RGB")

                        # Save converted image to a temporary JPEG
                        temp_path = image_path + ".tmp.jpg"

                        rgb_image.save(
                            temp_path,
                            format="JPEG",
                            quality=95
                        )

                        # Verify the converted image with TensorFlow
                        temp_data = tf.io.read_file(temp_path)

                        tf.io.decode_jpeg(
                            temp_data,
                            channels=3,
                            try_recover_truncated=False
                        ).numpy()

                        os.replace(temp_path, image_path)

                        converted += 1

                    else:
                        already_rgb += 1

            except Exception as error:

                print("\nInvalid image:", image_path)
                print("Reason:", error)

                # Remove temporary file if conversion failed
                temp_path = image_path + ".tmp.jpg"

                if os.path.exists(temp_path):
                    os.remove(temp_path)

                # Delete the invalid original image
                try:
                    os.remove(image_path)
                    deleted += 1
                    print("Corrupted image deleted.")

                except OSError as delete_error:
                    failed += 1
                    print("Could not delete:", delete_error)

    print("\nDATASET CLEANING SUMMARY")
    print("Images checked:", total_checked)
    print("Already RGB:", already_rgb)
    print("Converted to RGB:", converted)
    print("Corrupted images deleted:", deleted)
    print("Failed operations:", failed)


# ==================================================
# 6. FINAL TENSORFLOW VALIDATION
# ==================================================

def validate_dataset():

    print("\nFINAL DATASET VALIDATION")

    total_images = 0
    corrupted = 0
    non_rgb = 0

    for folder in dataset_folders:

        folder_path = os.path.join(DATASET_PATH, folder)

        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):

            if not filename.lower().endswith((".jpg", ".jpeg")):
                continue

            image_path = os.path.join(folder_path, filename)
            total_images += 1

            try:

                image_data = tf.io.read_file(image_path)

                tf.io.decode_jpeg(
                    image_data,
                    channels=3,
                    try_recover_truncated=False
                ).numpy()

                with Image.open(image_path) as image:

                    if image.mode != "RGB":
                        non_rgb += 1
                        print("Non-RGB image:", image_path)

            except Exception:
                corrupted += 1
                print("Invalid image:", image_path)

    print("\nVALIDATION SUMMARY")
    print("Total images checked:", total_images)
    print("Corrupted images:", corrupted)
    print("Non-RGB images:", non_rgb)


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(
            f"Dataset ZIP not found: {ZIP_PATH}"
        )

    analyze_dataset()

    analyze_dimensions()

    remove_corrupted_from_zip()

    demonstrate_preprocessing()

    clean_prepared_dataset()

    validate_dataset()
