import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Defining the path to the model and evaluation dataset
model_path = "Model/model.keras"
eval_data_path = "Data/fashion-mnist_test.csv"
output_path = "output.txt"

# Checking if the model file exists
if not os.path.exists(model_path):
    sys.exit("Error: Model file not found. Please provide a valid model path.")

# Checking if the evaluation dataset file exists
if not os.path.exists(eval_data_path):
    sys.exit("Error: Evaluation dataset file not found. Please provide a valid dataset path.")

# Loading the pre-trained model
model = tf.keras.models.load_model(model_path)

# Loading the evaluation dataset
eval_data = pd.read_csv(eval_data_path)
X_eval = eval_data.drop("label", axis=1).values
y_eval = eval_data["label"].values

X_eval = X_eval.reshape(-1, 28, 28, 1)

# Making predictions
y_pred = model.predict(X_eval)
y_pred_classes = np.argmax(y_pred, axis=1)

# Ensuring that y_pred_classes contains discrete class labels (integers)
y_pred_classes = np.array(y_pred_classes)

# Calculating accuracy, precision and recall
accuracy = accuracy_score(y_eval, y_pred_classes)
precision = precision_score(y_eval, y_pred_classes, average="macro")
recall = recall_score(y_eval, y_pred_classes, average="macro")

# Generating a classification report 
classification_rep = classification_report(y_eval, y_pred_classes)


# Writing the results to an output file
with open(output_path, "w") as f:
    f.write("Model's architecture summary:\n")
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.write(f"Evaluation accuracy: {accuracy:.3f}\n")
    f.write(f"Evaluation precision: {precision:.3f}\n")
    f.write(f"Evaluation recall: {recall:.3f}\n")
    f.write("\nClassification Report\n")
    f.write(classification_rep)

print("Evaluation completed. Results saved to output.txt.")