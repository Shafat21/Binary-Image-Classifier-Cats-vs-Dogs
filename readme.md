# Binary Image Classifier: Cats vs Dogs 🐱🐶

This project demonstrates the implementation of both a **Support Vector Machine (SVM)** classifier and a **Convolutional Neural Network (CNN)** for binary image classification (Cats vs Dogs). It includes data preprocessing, model training, evaluation, and prediction.

[![image.png](https://i.postimg.cc/MpZfXHYz/image.png)](https://postimg.cc/1fT3j9Td)

---

## Project Overview

- **Languages/Frameworks**: Python, NumPy, scikit-learn, TensorFlow, Keras, Matplotlib
- **Goal**: Classify images of cats and dogs using two approaches:
  - SVM with flattened grayscale images
  - CNN with augmented image data

---

## Features

1. **Image Preprocessing**:
   - Convert images to grayscale
   - Resize images to 128x128
   - Normalize pixel values
   - Flatten images for SVM

2. **Model Training**:
   - **SVM**: A linear kernel classifier for binary classification.
   - **CNN**: Deep learning model with multiple convolutional, pooling, and dense layers.

3. **Data Augmentation**:
   - Performed for CNN training to enhance model generalization using Keras' `ImageDataGenerator`.

4. **Evaluation Metrics**:
   - Accuracy, Precision, Recall, F1-Score (from classification report)
   - Confusion matrix visualization

5. **Visualization**:
   - Training and validation accuracy/loss graphs
   - Display predictions on test images

---

## Installation

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure your dataset is organized as follows:
   ```
   train2/
   ├── cat/
   │   ├── cat.1.jpg
   │   ├── cat.2.jpg
   │   └── ...
   └── dog/
       ├── dog.1.jpg
       ├── dog.2.jpg
       └── ...
   ```

---

## How to Run

1. **Preprocess Images**:
   Modify the `image_paths` variable in the script to point to your dataset directory.

2. **Train SVM**:
   Run the script to train and evaluate the SVM model.

3. **Train CNN**:
   The script will automatically train the CNN model and save training/validation accuracy and loss graphs.

4. **Predict**:
   Replace the `image_path` in the script with a test image path to generate predictions from both SVM and CNN.

---

## Results

### SVM Results
- **Classification Report**:
[![image.png](https://i.postimg.cc/YqRhckBZ/image.png)](https://postimg.cc/ykJV9Mdy)
- **Confusion Matrix**:
[![image.png](https://i.postimg.cc/zGPvBjBk/image.png)](https://postimg.cc/dhrwNGG7)

### CNN Results
- Training and validation accuracy/loss graphs:
[![image.png](https://i.postimg.cc/4xcq744F/image.png)](https://postimg.cc/zyJ04rgn)
[![image.png](https://i.postimg.cc/DzKDqV8f/image.png)](https://postimg.cc/jDZZbFHB)
- Example Prediction:
[![image.png](https://i.postimg.cc/MpZfXHYz/image.png)](https://postimg.cc/1fT3j9Td)


---

## Dependencies

- Python 3.x
- `numpy`
- `scikit-learn`
- `tensorflow`
- `keras`
- `matplotlib`
- `scikit-image`
- `tqdm`
