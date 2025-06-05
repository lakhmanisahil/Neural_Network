from cifar_utils import load_cifar10_data
from model import NeuralNet
from evaluate import evaluate
from plot_loss import plot_loss
import numpy as np

# Load data
X_train, Y_train, X_test, Y_test = load_cifar10_data()

# Initialize model
model = NeuralNet()

# Train
loss_history = model.train(X_train, Y_train, epochs=100)

# Evaluate
Y_pred = model.predict(X_test)
acc, prec, rec, f1, conf = evaluate(Y_test, Y_pred)

# Print metrics
print(f"Accuracy: {acc:.3f}")
for i in range(3):
    print(f"Class {i}: Precision={prec[i]:.3f}, Recall={rec[i]:.3f}, F1={f1[i]:.3f}")
print("Confusion Matrix:\n", conf)

# Plot training loss
plot_loss(loss_history)

