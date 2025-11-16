import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def one_hot_encode(labels, n_classes):
    one_hot = np.zeros((len(labels), n_classes))
    for i, label in enumerate(labels):
        one_hot[i][label] = 1
    return one_hot

class DataProcessor:
    def __init__(self, filepath="penguins.csv"):
        data = pd.read_csv(filepath)
        target = data["Species"]
        data = data.drop("Species", axis=1)

        numeric_data = data.select_dtypes(include=["number"])
        categorical_data = data.drop(columns=numeric_data.columns)
        numeric_data = numeric_data.fillna(numeric_data.mean())
        categorical_data = categorical_data.fillna(categorical_data.mode().iloc[0])

        data = pd.concat([numeric_data, categorical_data], axis=1)

        encoder = LabelEncoder()
        data["OriginLocation"] = encoder.fit_transform(data["OriginLocation"])
        data["Species"] = target
        
        self.data = data
        self.label_encoder_species = LabelEncoder()
        self.feature_columns = ["CulmenLength", "CulmenDepth", "FlipperLength", "OriginLocation", "BodyMass"]
        self.scaler = StandardScaler()

    def get_processed_data(self):
        self.data['Species_encoded'] = self.label_encoder_species.fit_transform(self.data['Species'])
        
        class_data = []
        for class_id in range(len(self.label_encoder_species.classes_)):
            class_samples = self.data[self.data['Species_encoded'] == class_id].copy()
            class_data.append(class_samples)

        train_data = []
        test_data = []

        for class_samples in class_data:
            train_data.append(class_samples.iloc[:30])
            test_data.append(class_samples.iloc[30:50])

        train_df = pd.concat(train_data, axis=0).reset_index(drop=True)
        test_df = pd.concat(test_data, axis=0).reset_index(drop=True)

        X_train = train_df[self.feature_columns].values
        X_test = test_df[self.feature_columns].values
        y_train_labels = train_df['Species_encoded'].values
        y_test = test_df['Species_encoded'].values

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        y_train = one_hot_encode(y_train_labels, len(self.label_encoder_species.classes_))

        return X_train, X_test, y_train, y_test

class MLP:
    def __init__(self, layer_sizes, learning_rate, epochs, activation_func, use_bias):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.use_bias = use_bias
        self.n_classes = layer_sizes[-1]

        if activation_func == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        else:
            self.activation = tanh
            self.activation_derivative = tanh_derivative

        self.weights = []
        self.biases = []

        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            self.weights.append(w)

            if use_bias:
                b = np.random.randn(layer_sizes[i+1]) * 0.1
                self.biases.append(b)
            else:
                self.biases.append(np.zeros(layer_sizes[i+1]))

    def forward(self, X):
        activations = [X]
        net_values = []
        current_activation = X

        for i in range(len(self.weights)):
            net = np.dot(current_activation, self.weights[i])
            if self.use_bias:
                net += self.biases[i]
            
            net_values.append(net)
            current_activation = self.activation(net)
            activations.append(current_activation)

        return activations, net_values

    def backward(self, X, y, activations, net_values):
        n_layers = len(self.weights)
        errors = [None] * n_layers

        output_error = (y - activations[-1]) * self.activation_derivative(net_values[-1])
        errors[-1] = output_error

        for i in range(n_layers - 2, -1, -1):
            error = np.dot(errors[i + 1], self.weights[i + 1].T) * self.activation_derivative(net_values[i])
            errors[i] = error

        for i in range(n_layers):
            weight_update = self.learning_rate * np.outer(activations[i], errors[i])
            self.weights[i] += weight_update

            if self.use_bias:
                bias_update = self.learning_rate * errors[i]
                self.biases[i] += bias_update

    def fit(self, X_train, y_train):
        for _ in range(self.epochs):
            for i in range(len(X_train)):
                activations, net_values = self.forward(X_train[i])
                self.backward(X_train[i], y_train[i], activations, net_values)

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            activations, _ = self.forward(X[i])
            output = activations[-1]
            predicted_class = np.argmax(output)
            predictions.append(predicted_class)
        return np.array(predictions)

    def evaluate(self, X, y_true):
        predictions = self.predict(X)
        
        confusion_matrix_data = np.zeros((self.n_classes, self.n_classes), dtype=int)
        
        for true_label, pred_label in zip(y_true, predictions):
            confusion_matrix_data[true_label][pred_label] += 1

        correct = np.sum(predictions == y_true)
        accuracy = (correct / len(y_true))
        
        return accuracy, confusion_matrix_data