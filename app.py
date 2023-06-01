from flask import Flask, request, jsonify
import numpy as np
import cv2
import urllib.request
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Set the maximum content length to 10MB

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='D:\wep APP 2\wep APP\model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file was uploaded
    if not request.json or 'file' not in request.json[0]:
        return jsonify({'error': 'No file URL provided'}), 400

    file_url = request.json[0]['file']
    try:
        # Download the image from the URL
        file = urllib.request.urlopen(file_url).read()
        img = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
    except:
        return jsonify({'error': 'Failed to download or read the image'}), 400

    # Check if the image size is too large
    if img.size > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({'error': 'Image is too large'}), 400

    # Preprocess the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # Set the input tensor
    input_data = np.array(img, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run the inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred = np.argmax(output_data, axis=1)

    # Return the prediction as a string
    if pred == 0:
        return jsonify({'prediction': 'glioma_tumor'}), 200
    elif pred == 1:
        return jsonify({'prediction': 'no_tumor'}), 200
    elif pred == 2:
        return jsonify({'prediction': 'meningioma_tumor'}), 200
    else:
        return jsonify({'prediction': 'pituitary_tumor'}), 200

if __name__ == '__main__':
    app.run(debug=True)
