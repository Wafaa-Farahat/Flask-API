import cv2
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np

app = Flask(__name__)


modelZheimar = keras.models.load_model('Alzheimer.h5')  # type: ignore


@app.route('/api/classify2/', methods=['POST'])

def classifyZheimar_image():
    # load image from request
    imageZheimar = request.files['image'].read()

    # preprocess image
    imageZheimar = preprocessZheimar_image(imageZheimar)

    # run inference on TensorFlow model
    resultZheimar = modelZheimar.predict(imageZheimar)

    # format the results

# Define a list of labels corresponding to different classes    
    labelsZheimar = ['MildDemented', 'ModerateDemented',
              'NonDemented', 'VeryMildDemented']
    # Initialize an empty dictionary to store the results for each class
    ResultZheimar = {}
# Initialize variables to keep track of the maximum probability and corresponding label
    maxZheimar_prob = 0.0

    maxZheimar_label = ''

    # Iterate over the labels using the enumerate function, which provides both the index and the value
    for i, labelZheimar in enumerate(labelsZheimar):
         # Access the probability of the current class from the prediction results
        probZheimar = float(resultZheimar[0][i])
        # Store the probability in the ResultZheimar dictionary with the corresponding label
        ResultZheimar[labelZheimar] = probZheimar
        # Check if the current probability is greater than the maximum probability
        if probZheimar > maxZheimar_prob:
          # If true, update the maximum probability and corresponding label
          maxZheimar_prob = probZheimar
          maxZheimar_label = labelZheimar

# print the label with the highest probability
    print("Most likely label: ", maxZheimar_label)
    accZheimar = " 99.45 %"
    #   return jsonify(Result)

    return jsonify(ResultZheimar={'  Diagnose': maxZheimar_label,  'accuracy': accZheimar})

    # return "hello"


def preprocessZheimar_image(imageZheimar):
    # preprocess the imageZheimar (resize, normalize, etc.)
    print(imageZheimar)
    imageZheimar = np.asarray(bytearray(imageZheimar), dtype="uint8")
    imageZheimar = cv2.imdecode(imageZheimar, cv2.IMREAD_COLOR)

    # n_img = cv2.imread(imageZheimar, cv2.IMREAD_COLOR)
    n_imgZheimar_size = cv2.resize(imageZheimar, (224, 224), interpolation=cv2.INTER_LINEAR)

    return np.array([n_imgZheimar_size])



if __name__ == 'main':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000, host='0.0.0.0')
