# Cats vs Dogs Classification using MobileNetV2

A deep learning image classification project that uses **Transfer Learning with MobileNetV2** to classify images into two categories: Cats and Dogs.

The project implements a pretrained MobileNetV2 model, trains a custom classification head, and further improves the model through fine-tuning. Both models are evaluated using accuracy, loss, confusion matrix, precision, recall and F1-score.

## Project Overview

The objective of this project is to build an image classification model capable of distinguishing between cats and dogs.

Instead of training a Convolutional Neural Network from scratch, this project uses **MobileNetV2 pretrained on ImageNet** as the feature extraction backbone.

The implementation consists of two training stages:

1. **Feature Extraction:** The pretrained MobileNetV2 base is frozen, and a custom classification head is trained.
2. **Fine-Tuning:** The last 30 layers of the pretrained backbone are unfrozen and trained with a low learning rate to adapt the learned features to the Cats vs Dogs dataset.

## Model Architecture

The model uses the following architecture:

| Layer | Configuration |
|---|---|
| Base Model | MobileNetV2 |
| Pretrained Weights | ImageNet |
| Input Shape | 224 × 224 × 3 |
| Global Pooling | GlobalAveragePooling2D |
| Regularization | Dropout (0.2) |
| Output Layer | Dense (2, Softmax) |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |

### Training Configuration

**Stage 1: Feature Extraction**

- MobileNetV2 base layers frozen.
- Only the classification head is trainable.
- 10 training epochs.
- Batch size: 32.
- Learning rate: Adam default.

**Stage 2: Fine-Tuning**

- Last 30 MobileNetV2 layers made trainable.
- Batch Normalization layers kept frozen.
- Learning rate: 0.00001.
- Maximum 5 epochs.
- Early stopping based on validation loss.
- Best weights restored when training stops.

## Dataset

The project uses the Microsoft Cats vs Dogs dataset.

Original dataset:
- 12,500 Cat images.
- 12,500 Dog images.
- 25,000 images in total.

Two corrupted images were identified during dataset inspection and excluded from the prepared dataset.

The final prepared dataset used for the reported evaluation contained:

| Split | Number of Images |
|---|---:|
| Training | 17,345 |
| Validation | 3,707 |
| Testing | 3,721 |
| **Total** | **24,773** |

The dataset was divided into training, validation and testing subsets using stratified splitting.

## Dataset Download

This project uses the [Microsoft Cats vs Dogs Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765).

**Dataset details:**
- Total images: 25,000
- Cat images: 12,500
- Dog images: 12,500
- Download size: Approximately 786.7 MB

### Instructions

1. Download the dataset from the [official Microsoft download page](https://www.microsoft.com/en-us/download/details.aspx?id=54765).
2. Extract the downloaded ZIP file.
3. Place the `PetImages` folder inside the `dataset` directory.
4. Run `eda.py` to inspect and clean the images.
5. Run `prepare_dataset.py` to create the training, validation and testing splits.

**Note:** The dataset is not included in this GitHub repository because of its size. Download it separately before running the training pipeline.
## Data Preprocessing

The following preprocessing operations were performed:

1. Image inspection and corrupted-image detection.
2. Conversion of non-RGB images into RGB format.
3. Resizing images to 224 × 224 pixels.
4. Conversion of images into NumPy arrays.
5. Application of MobileNetV2's `preprocess_input()` function.
6. Conversion of pixel values from the original 0–255 range to the expected -1 to 1 range.

Images were loaded using TensorFlow's `image_dataset_from_directory()`.

## Model Evaluation

Both the original model and the fine-tuned model were evaluated on the same test dataset containing 3,721 images.

### 1. Original MobileNetV2 Model

| Metric | Result |
|---|---:|
| Test Accuracy | 99.1669% |
| Test Loss | 0.02462 |

**Confusion Matrix**

```text
              Predicted
              Cat    Dog

Actual Cat   1846     15
Actual Dog     16   1844
```

**Classification Report**

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Cat | 0.99 | 0.99 | 0.99 | 1,861 |
| Dog | 0.99 | 0.99 | 0.99 | 1,860 |

### 2. Fine-Tuned MobileNetV2 Model

| Metric | Result |
|---|---:|
| Test Accuracy | 99.1938% |
| Test Loss | 0.02560 |

**Confusion Matrix**

```text
              Predicted
              Cat    Dog

Actual Cat   1852      9
Actual Dog     21   1839
```

**Classification Report**

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Cat | 0.99 | 1.00 | 0.99 | 1,861 |
| Dog | 1.00 | 0.99 | 0.99 | 1,860 |

### 3. Model Comparison

| Metric | Original Model | Fine-Tuned Model |
|---|---:|---:|
| Test Accuracy | 99.1669% | 99.1938% |
| Test Loss | 0.02462 | 0.02560 |
| Correct Predictions | 3,690 | 3,691 |
| Incorrect Predictions | 31 | 30 |

Fine-tuning improved test accuracy by approximately **0.0269 percentage points**, reducing the number of incorrect predictions from 31 to 30.

However, test loss increased slightly from 0.02462 to 0.02560. This means fine-tuning did not improve every evaluation metric.

The improvement is modest, and the original model already performed very well.

## Project Structure

```text
Cats_Dogs_Classification/
│
├── dataset/
│   ├── PetImages/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── cats_and_dogs_mobilenetv2.keras
├── cats_dogs_finetuned.keras
│
├── eda.py
├── prepare_dataset.py
├── load_dataset.py
├── model.py
├── train.py
├── finetune.py
├── evaluate.py
├── finaltesting.py
└── inspect_errors.py
```

### Python Files

| File | Description |
|---|---|
| `eda.py` | Dataset inspection, image analysis, RGB conversion and image validation. |
| `prepare_dataset.py` | Splits the dataset into training, validation and testing sets. |
| `load_dataset.py` | Loads datasets and applies MobileNetV2 preprocessing. |
| `model.py` | Defines the MobileNetV2 classification architecture. |
| `train.py` | Trains the initial model using transfer learning. |
| `finetune.py` | Fine-tunes the pretrained MobileNetV2 backbone. |
| `evaluate.py` | Evaluates the original model and generates classification metrics. |
| `finaltesting.py` | Evaluates the fine-tuned model. |
| `inspect_errors.py` | Inspects incorrectly classified test images. |

## Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Pandas
- Matplotlib
- Pillow
- Scikit-learn

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/cats-dogs-classification.git

cd cats-dogs-classification
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn pillow
```

Use a compatible TensorFlow version for the saved `.keras` model.

### 4. Prepare the dataset

Place the original dataset in the expected directory structure and run:

```bash
python eda.py
python prepare_dataset.py
```

### 5. Train the model

```bash
python train.py
```

### 6. Fine-tune the model

```bash
python finetune.py
```

### 7. Evaluate the model

```bash
python evaluate.py
python finaltesting.py
```

**Note:** The repository may contain pretrained model files. If those files are included, you can evaluate the existing models without training them again.

## Limitations

- The model is trained specifically for binary Cat vs Dog classification.
- It is not designed to classify other animal species or arbitrary objects.
- Performance on images outside the training dataset may differ.
- The reported results are based on one test split and do not guarantee equivalent performance on every real-world image.
- High confidence does not necessarily mean a prediction is correct.
- Fine-tuning produced only a marginal accuracy improvement in this experiment.

## Future Improvements

- Develop a web application for image uploads and real-time predictions.
- Deploy the model using Streamlit or another hosting platform.
- Add prediction confidence visualization.
- Test performance on external images not included in the original dataset.
- Explore data augmentation and additional model architectures.

## Author

**Akshat**

Developed as a practical deep learning project to understand transfer learning, model fine-tuning, image preprocessing and evaluation using TensorFlow and MobileNetV2.
