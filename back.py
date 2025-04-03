from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("model.h5")

app = Flask(__name__)

# Route for the homepage (index)
@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Digit Prediction</title>
    </head>
    <body>
      <h1>Upload an Image of a Digit</h1>
      <form action="/predict" method="POST" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">Predict</button>
      </form>
    </body>
    </html>
    '''

# Route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'File not found in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Process the uploaded image
        image = Image.open(file).convert('L').resize((28, 28))
        image_array = np.array(image) / 255.0
        image_array = image_array.reshape(1, 28, 28)

        # Predict the digit
        prediction = model.predict(image_array)
        predicted_class = int(np.argmax(prediction))

        return jsonify({'Predicted Digit': predicted_class})
    except Exception as e:
        print(f"Error processing file: {e}")
        return jsonify({'error': 'Error processing image'}), 500


if __name__ == "__main__":
    app.run(debug=True)
