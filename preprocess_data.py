import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Define paths for dataset directories
free_parking_dir = 'data/free/'
occupied_parking_dir = 'data/occupied/'

# Initial parameters
IMG_SIZE = 150  # resize images to 150 * 150 pixels for better performance‚
data = []
labels = []

# Method to preprocess images
def preprocess_image(directory, label):
    """
    """
    for image_name in os.listdir(directory):
        # Obtain the image path
        image_path = os.path.join(directory, image_name)

        # Read the image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            # resize images to 150*150 pixels
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            # Append the processed image to the data list.
            data.append(img)
            # Append the labels o or 1 to the labels list
            labels.append(label)

# Prepricess free image with label 0
preprocess_image(free_parking_dir, 0)

# preprocess occupied image with label 1s
preprocess_image(occupied_parking_dir, 1)

# Convert data and labels to NumPy arrays
data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # Reshape the data array to include the channel dimension
labels = np.array(labels)  # Convert labels list to a NumPy array

# Normalize pixels to the range [0,1]
data = data / 255.0

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Save the preprocessed data
np.savez('data/preprocessed_data.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)




