# Digit Recognizer

A Convolutional Neural Network (CNN) solution for the Kaggle Digit Recognizer competition.

## Approach

The notebook uses TensorFlow/Keras to classify handwritten digits from 28×28 pixel images.

### Preprocessing
- Separates the `label` column from the training data
- Reshapes images to `28 × 28 × 1`
- Normalizes pixel values to the range 0–1
- One-hot encodes the digit labels

### CNN Architecture
- Conv2D — 32 filters, 3×3, ReLU
- MaxPooling2D — 2×2
- Conv2D — 32 filters, 3×3, ReLU
- MaxPooling2D — 2×2
- Conv2D — 16 filters, 3×3, ReLU
- Flatten
- Dense — 10 outputs, Softmax

The model is trained using the Adam optimizer and categorical cross-entropy loss for 10 epochs, with 20% of the training data used for validation and a batch size of 128.

## Submission

Predictions are generated for the test set and saved as:

`submission.csv`

The submission contains the required `ImageId` and `label` columns.

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib