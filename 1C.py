
# 1. Import necessary packages

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split


# 2. Load CIFAR-10 CSV dataset

data = pd.read_csv("cifar-10.csv")

print("Dataset Shape:", data.shape)

# Separate features and labels

X = data.drop("label", axis=1).values
y = data["label"].values

# Normalize pixel values

X = X / 255.0

# First split: Training and Testing

x_train_full, x_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Second split: Training and Validation

x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full,
    test_size=0.15,
    random_state=42,
    stratify=y_train_full
)

print("Training:", x_train.shape)
print("Validation:", x_val.shape)
print("Testing:", x_test.shape)


# 3. Define FNN Architecture

model = Sequential([

    Dense(1024, activation='relu',
          input_shape=(3072,)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(256, activation='relu'),
    Dropout(0.2),

    Dense(10, activation='softmax')

])

model.summary()


# 4. Compile the model using Adam optimizer

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# 5. Define callbacks

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001
)


# 6. Train the model

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=40,
    batch_size=128,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)


# 7. Evaluate the network

test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy * 100, "%")


# 8. Plot training accuracy and loss

plt.figure(figsize=(12, 5))

# Accuracy graph

plt.subplot(1, 2, 1)

plt.plot(history.history['accuracy'],
         label='Training Accuracy')

plt.plot(history.history['val_accuracy'],
         label='Validation Accuracy')

plt.title('FNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)


# Loss graph

plt.subplot(1, 2, 2)

plt.plot(history.history['loss'],
         label='Training Loss')

plt.plot(history.history['val_loss'],
         label='Validation Loss')

plt.title('FNN Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# 9. Predict sample images

class_names = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

predictions = model.predict(x_test[:5])

for i in range(5):

    predicted_class = np.argmax(predictions[i])

    # Display the image.
    # Assumes CSV pixels are stored in RGB image order.
    image = x_test[i].reshape(32, 32, 3)

    plt.figure(figsize=(3, 3))
    plt.imshow(image)

    plt.title(
        f"Predicted: {class_names[predicted_class]}\n"
        f"Actual: {class_names[y_test[i]]}"
    )

    plt.axis('off')
    plt.show()
