from flask import Flask, request, jsonify, send_from_directory
from flask import render_template
import numpy as np
import tensorflow as tf
import os
import cv2

# Initialize the Flask app
app = Flask(__name__)

# ensure the uploads file directory exists
uploads_folder ='uploads/'
if not os.path.exists(uploads_folder):
    os.makedirs(uploads_folder)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to serve the favicon
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(directory='static', filename='favicon.ico')

# Load the trained model
model = tf.keras.models.load_model('models/parking_space_model.keras')

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle POST requests for predictions.
    Expects JSON data with a key 'input' that contains the input data.
    """
    # Check if an image is uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file part'}), 400

    # Save the file to a temporary location
    filepath = os.path.join(uploads_folder, file.filename)
    file.save(filepath)

    # Preprocess the image
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    # resize the image
    img = cv2.resize(img, (150, 150))

    # normalize the pixel value
    img = img/255.0

    # reshape the image
    img = img.reshape(1,150,150,1)

    # Make the prediction
    prediction = model.predict(img)
    print(f"Raw prediction value: {prediction[0][0]}")  # Debugging
    result = "Free" if prediction[0][0] < 0.5 else "Occupied"
    print(f"Prediction: {result}")  # This will print to the console

    #return the result as a part of html page
    return render_template('result.html', prediction=result)

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)