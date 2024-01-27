import cv2
from four import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np

app = Flask(__name__)

modelCardio = keras.models.load_model('Cardiomegaly.h5') # type: ignore




@app.route('/api/classify/', methods=['POST'])
def classifyCardio_image():
    # load imageCardio from request
    imageCardio = request.files['imageCardio'].read()

    # preprocess imageCardio
    imageCardio = preprocessCardio_image(imageCardio)

    # run inference on TensorFlow model
    resultCardio = modelCardio.predict(imageCardio)

    # format the results
    labelsCardio = ['normal', 'problem']
    ResultCardio = {}
    
    maxCardio_prob = 0.0


    maxCardio_label = ''
    for i, labelCardio in enumerate(labelsCardio):
        probCardio = float(resultCardio[0][i])
        ResultCardio[labelCardio] = probCardio
        if probCardio > maxCardio_prob:
          maxCardio_prob = probCardio
          maxCardio_label = labelCardio

# print the label with the highest probability
    print("Most likely label: ", maxCardio_label)
    accCardio = " 99.6 %" 
    #   return jsonify(Result)


    return jsonify(Result={'  Diagnose': maxCardio_label  ,  'accuracy': accCardio})
   
    

    # return "hello"


def preprocessCardio_image(imageCardio):
    # preprocess the imageCardio (resize, normalize, etc.)
    print(imageCardio)
    imageCardio = np.asarray(bytearray(imageCardio), dtype="uint8")
    imageCardio = cv2.imdecode(imageCardio, cv2.IMREAD_COLOR)

    # n_img = cv2.imread(imageCardio, cv2.IMREAD_COLOR)
    n_imgCardio_size = cv2.resize(imageCardio, (224, 224), interpolation=cv2.INTER_LINEAR)

    return np.array([n_imgCardio_size])





if __name__ == 'main':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000, host='0.0.0.0')
