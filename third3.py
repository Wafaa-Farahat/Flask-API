import cv2
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np

app = Flask(__name__)

modelGastro = keras.models.load_model('Gastric_version_2.h5')  # type: ignore



@app.route('/api/classifyGastro/', methods=['POST'])
def classifyGastro_image():
    # load imageCardio from request
    imageGastro = request.files['imageGastro'].read()

    # preprocess imageCardio
    imageGastro = preprocessGastro_image(imageGastro)

    # run inference on TensorFlow model
    resultGastro = modelGastro.predict(imageGastro)

    # format the results
    labelsGastro = ['MSIMUT', 'MSS']
    ResultGastro = {}
    
    maxGastro_prob = 0.0


    maxGastro_label = ''
    for i, labelGastro in enumerate(labelsGastro):
        probGastro = float(resultGastro[0][i])
        ResultGastro[labelGastro] = probGastro
        if probGastro > maxGastro_prob:
          maxGastro_prob = probGastro
          maxGastro_label = labelGastro

# print the label with the highest probability
    print("Most likely label: ", maxGastro_label)
    accGastro = " 99.2 %" 
    #   return jsonify(Result)


    return jsonify(Result={'  Diagnose': maxGastro_label  ,  'accuracy': accGastro})
   
    

    # return "hello"


def preprocessGastro_image(imageGastro):
    # preprocess the imageGastro (resize, normalize, etc.)
    print(imageGastro)
    imageGastro = np.asarray(bytearray(imageGastro), dtype="uint8")
    imageGastro = cv2.imdecode(imageGastro, cv2.IMREAD_COLOR)

    # n_img = cv2.imread(imageGastro, cv2.IMREAD_COLOR)
    n_imgGastro_size = cv2.resize(imageGastro, (65, 65), interpolation=cv2.INTER_LINEAR)

    return np.array([n_imgGastro_size])





if __name__ == 'main':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000, host='0.0.0.0')
