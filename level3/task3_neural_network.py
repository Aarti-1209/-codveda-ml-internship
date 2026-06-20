"""
Codveda Technologies - Machine Learning Internship
Level 3 (Advanced) - Task 3: Neural Networks with TensorFlow/Keras

Description: Build a simple feed-forward neural network using
TensorFlow/Keras to classify Iris flower species.

Objectives:
1. Load and preprocess the dataset
2. Design a neural network architecture (input, hidden, output layers)
3. Train the model using backpropagation
4. Evaluate the model using accuracy and visualize training/validation loss

Tools: Python, TensorFlow/Keras, pandas, matplotlib
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)

# ---------------------------------------------------------
# 1. LOAD AND PREPROCESS THE DATASET
# ---------------------------------------------------------
df = pd.read_csv("../data/iris.csv")
print("Dataset shape:", df.shape)

X = df.drop(columns=['species']).values
y_raw = df['species'].values

# Encode string labels to integers, then one-hot encode for multi-class output
le = LabelEncoder()
y_int = le.fit_transform(y_raw)
y_onehot = keras.utils.to_categorical(y_int, num_classes=3)
print("Classes:", le.classes_)

# Scale features (neural nets train much better on standardized inputs)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_onehot, test_size=0.2, random_state=42, stratify=y_int
)

print("Train shape:", X_train.shape, "| Test shape:", X_test.shape)

# ---------------------------------------------------------
# 2. DESIGN THE NEURAL NETWORK ARCHITECTURE
# ---------------------------------------------------------
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(16, activation='relu', name='hidden_layer_1'),
    layers.Dense(8, activation='relu', name='hidden_layer_2'),
    layers.Dense(3, activation='softmax', name='output_layer')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------------------------------------------------
# 3. TRAIN THE MODEL USING BACKPROPAGATION
# ---------------------------------------------------------
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.2,
    verbose=0
)

print("\nTraining complete.")
print(f"Final training accuracy:   {history.history['accuracy'][-1]:.3f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.3f}")

# ---------------------------------------------------------
# 4. EVALUATE THE MODEL
# ---------------------------------------------------------
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.3f}")
print(f"Test Accuracy: {test_accuracy:.3f}")

# ---------------------------------------------------------
# 5. VISUALIZE TRAINING / VALIDATION LOSS AND ACCURACY
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(history.history['loss'], label='Training Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training vs Validation Loss')
axes[0].legend()

axes[1].plot(history.history['accuracy'], label='Training Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training vs Validation Accuracy')
axes[1].legend()

plt.tight_layout()
plt.savefig("neural_network_training_curves.png", dpi=150)
print("\nTraining curves saved as neural_network_training_curves.png")
