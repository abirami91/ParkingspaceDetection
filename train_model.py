import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau  # Import the callback

# Define the directory
directory = 'models'

# Check if the directory exists, if not, create it
if not os.path.exists(directory):
    os.makedirs(directory)

# Load preprocessed data
data = np.load('data/preprocessed_data.npz')
X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

# Define data augmentation strategy
# This will improve the existing overfitting issue due to less data.
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the data generator on your training data
datagen.fit(X_train)

# Use the CNN model(sequential to train the data)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # Add dropout here
    Dense(1, activation='sigmoid')
])

# Use an Adam optimizer to check the loss and accuracy rate 
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Define the learning rate scheduler callback
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train the model using the augmented data with the learning rate scheduler
model.fit(datagen.flow(X_train, y_train, batch_size=2), epochs=10, validation_data=(X_test, y_test), callbacks=[lr_scheduler])

# Save the model
model.save(os.path.join(directory, 'parking_space_model.keras'))