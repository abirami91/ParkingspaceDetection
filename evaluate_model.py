import numpy as np
import tensorflow as tf
# Optionally visualize some predictions
import matplotlib.pyplot as plt

# load the preprocessed data
data = np.load('data/preprocessed_data.npz')
X_test, y_test = data['X_test'], data['y_test']

# Load the trained model
model = tf.keras.models.load_model('models/parking_space_model.keras')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Make predictions
predictions = (model.predict(X_test) > 0.5).astype("int32")

# Simulate color by repeating the grayscale values across 3 channels
X_test_rgb = np.repeat(X_test, 3, axis=-1)

for i in range(7):
    plt.imshow(X_test_rgb[i].reshape(150, 150, 3))
    plt.title(f'Predicted: {"Free" if predictions[i] == 0 else "Occupied"}, Actual: {"Free" if y_test[i] == 0 else "Occupied"}')
    plt.show()