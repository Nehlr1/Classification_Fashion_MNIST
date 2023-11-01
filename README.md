# Classification Fashion MNIST

This repository contains code for performing image classification on the Fashion MNIST dataset using machine learning techniques (CNN).

## Dataset

The Fashion MNIST dataset is a collection of 70,000 grayscale images of 10 different fashion categories, with each image being a 28x28 pixel representation. The dataset is split into a training set of 60,000 images and a test set of 10,000 images.

- Downlaod the dataset:
  ```
  https://www.kaggle.com/datasets/zalando-research/fashionmnist/data
  ```

## Why CNN (Convolutional Neural Network) was used:
- CNNs are perfect for handling picture data because they can recognize features at many levels of abstraction, from edges to intricate textures, and they can record hierarchical spatial patterns. They use parameter sharing to reduce overfitting and enhance generalization, and they offer translation invariance, which is necessary for applications like picture classification. CNNs also automatically extract features, which eliminates the need for human feature engineering.

## Reasons for CNN Parameters:

- Input Shape (28x28x1): The input shape corresponds to the dimensions of the Fashion MNIST images (28x28 pixels with a single channel for grayscale). This shape is compatible with the dataset.

- Convolutional Layer (32 filters, 3x3 kernel, ReLU activation): Convolutional layers are responsible for feature extraction. The choice of 32 filters with a 3x3 kernel is to capture various patterns in the images. ReLU activation is used to introduce non-linearity.

- Max-Pooling Layer (2x2 pool size): Max-pooling layers reduce the spatial dimensions of the feature maps, aiding in translation invariance and computational efficiency.

- Flatten Layer: This layer reshapes the output from the previous layers into a 1D vector, preparing it for fully connected layers.

- Dropout (0.2): Dropout is used to prevent overfitting. A dropout rate of 0.2 means that 20% of the neurons are randomly dropped out during training, which helps in generalization.

- Dense Layers (128 neurons, ReLU activation): These fully linked layers use the retrieved characteristics to conduct categorization. The 128 neurons that were selected are fairly random and may be changed based on the task's difficulty and the available processing power. Non-linearity is introduced by ReLU activation.

- Output Layer (10 neurons, Softmax activation): The output layer has 10 neurons, matching the number of classes in Fashion MNIST. Softmax activation converts the final layer's raw scores into class probabilities.

## Code Files

- `classification.ipynb`: This Jupyter Notebook contains the code for training and evaluating the classification model on the Fashion MNIST dataset.
- `evaluate_model.py`: This Python script provides functions for evaluating the trained model on new test data.

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository: 
    ```
    git clone https://github.com/Nehlr1/Classification_Fashion_MNIST.git
    ```
2. Installing virtualenv:
   - For Windows:
     ```
     py -m pip install --user virtualenv
     ```
   - For macOS/Linux:
     ```
     python3 -m pip install --user virtualenv
     ```
2. Create a virtual environment:
   - For Windows:
     ```
     python -m venv venv
     venv\Scripts\activate
     ```
   - For macOS/Linux:
     ```
     python3 -m venv venv
     source venv/bin/activate
     ```
3. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```
4. Open and run the `classification.ipynb` notebook to train the classification model.
5. Open and run the `evaluate_model.py` to evaluate its performance by creating the output.txt file.

## Dependencies

The project has the following dependencies:

- NumPy (version 1.23.5)
- Pandas (version 2.0.2)
- Scikit Learn (version 1.1.3)
- TensorFlow (version 2.12.0)
- Matplotlib (version 3.5.3)

All the required dependencies are listed in the `requirements.txt` file.

## Results

After training the model, you can evaluate its performance using the `evaluate_model.py` script. The script will load the trained model and test it on new images from the test set, providing accuracy metrics and classification results.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## Tutorial Video

https://github.com/Nehlr1/Classification_Fashion_MNIST/assets/87631464/6b63f913-1fd5-4dbd-9307-ac9d798efcdf
