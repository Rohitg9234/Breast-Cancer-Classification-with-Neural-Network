# Breast Cancer Classification with Neural Network

This project implements a neural network for classifying breast cancer data as either benign or malignant using a deep learning model built with TensorFlow and Keras. The dataset used is from the `sklearn` library, which contains various features (such as radius, texture, perimeter, area, etc.) of cell nuclei from breast cancer biopsies.

## Project Overview

This project aims to train a neural network to predict whether a tumor is benign or malignant based on certain features. The code loads the breast cancer dataset, preprocesses the data, splits it into training and testing sets, and then trains a simple neural network model. Finally, it evaluates the model's performance and demonstrates how to use the trained model for making predictions on new data.

## Steps Covered in the Code

### 1. **Importing Dependencies**

First, necessary libraries are imported, including:
- `numpy`, `pandas`, and `matplotlib` for data manipulation and visualization.
- `sklearn.datasets` for loading the breast cancer dataset.
- `train_test_split` to split the dataset into training and testing sets.
- `StandardScaler` to standardize the feature data.
- `tensorflow` and `keras` for building and training the neural network model.

### 2. **Data Collection & Processing**

The breast cancer dataset is loaded using `sklearn.datasets.load_breast_cancer()` function. This dataset contains information on various features like mean radius, mean texture, and mean perimeter of the cells. The code performs several data processing steps:
- Convert the data into a pandas DataFrame for easier manipulation.
- Add the target variable (malignant or benign) to the DataFrame.
- Check for missing values using `isna()`.
- Display summary statistics and check the distribution of the target variable (malignant vs benign).

### 3. **Splitting the Data into Training and Testing**

The data is split into training and testing sets using the `train_test_split` function. 
- 80% of the data is used for training the model.
- 20% is reserved for testing the model’s performance after training.

The features (`x`) are standardized using `StandardScaler`, which scales them to have zero mean and unit variance. This step helps improve the performance of the neural network.

### 4. **Building the Neural Network**

A simple feedforward neural network is built using Keras:
- **Input Layer**: The input layer flattens the 30 features (each representing a measurement) into a one-dimensional vector.
- **Hidden Layer**: A fully connected layer with 20 neurons and ReLU activation.
- **Output Layer**: A fully connected layer with 2 neurons (since the target variable has 2 classes: malignant and benign) and a sigmoid activation function, which is used for binary classification tasks.

### 5. **Compiling the Model**

The model is compiled with:
- `Adam` optimizer for faster convergence.
- `sparse_categorical_crossentropy` loss function, suitable for multi-class classification problems where the target variable is represented as integers (0 or 1 in this case).
- Accuracy is used as the evaluation metric.

### 6. **Training the Model**

The model is trained for 10 epochs (iterations) on the training data with a validation split of 10%. The history of training and validation accuracy and loss are stored for visualization.

### 7. **Model Evaluation**

After training, the model is evaluated on the test data using `model.evaluate()`, and the accuracy is printed. The model is tested on unseen data to check how well it generalizes.

### 8. **Making Predictions**

Finally, the trained model is used to predict whether a new sample is benign or malignant. A sample with 30 feature values is passed to the model, and the output prediction is made.

The code converts the model's prediction output to either 0 (benign) or 1 (malignant) based on the sigmoid output threshold of 0.5.

### 9. **Visualizing Results**

Two plots are created:
1. **Accuracy Plot**: Displays the training and validation accuracy over each epoch.
2. **Loss Plot**: Displays the training and validation loss over each epoch.

These plots help in understanding the model’s performance during training and if it’s overfitting or underfitting.

## Project Files

- `breast_cancer_classification_with_neural_network.ipynb`: The Jupyter Notebook with the full implementation of the code.
- Dependencies: `numpy`, `pandas`, `matplotlib`, `sklearn`, `tensorflow`, `keras`

## Requirements

Make sure the following libraries are installed:
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

## How to Run the Code

1. Clone or download the repository.
2. Open the Jupyter notebook `breast_cancer_classification_with_neural_network.ipynb`.
3. Run each cell to train the model and view the results.
4. The trained model can be used to predict new samples by passing the 30 feature values of a new sample to the `predict()` function.

## Example Input Data

Here’s an example of input data (features of a new tumor):
```python
input_data = (17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189)
```

The model will return either `Benign` or `Malignant` based on this input data.

## Conclusion

This project demonstrates how to use a simple neural network for a classification problem. It shows the importance of data preprocessing, model training, and evaluation. The model can predict whether a tumor is benign or malignant, helping in breast cancer diagnosis.

Let me know if you need any more details!
