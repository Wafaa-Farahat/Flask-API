import cv2
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np

app = Flask(__name__)

# Cardio model
modelCardio = keras.models.load_model('Cardiomegaly.h5')  # type: ignore

# Al zehimar model
modelZheimar = keras.models.load_model('Alzheimer.h5')  # type: ignore

# Gastro model
modelGastro = keras.models.load_model('Gastric_version_2.h5')  # type: ignore


# end point cardio


@app.route('/api/classifyCardio/', methods=['POST'])
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

    return jsonify(Result={'  Diagnose': maxCardio_label,  'accuracy': accCardio})

    # return "hello"


def preprocessCardio_image(imageCardio):
    # preprocess the imageCardio (resize, normalize, etc.)
    print(imageCardio)
    imageCardio = np.asarray(bytearray(imageCardio), dtype="uint8")
    imageCardio = cv2.imdecode(imageCardio, cv2.IMREAD_COLOR)

    # n_img = cv2.imread(imageCardio, cv2.IMREAD_COLOR)
    n_imgCardio_size = cv2.resize(
        imageCardio, (224, 224), interpolation=cv2.INTER_LINEAR)

    return np.array([n_imgCardio_size])


# end point Al zehimar


@app.route('/api/classifyZehimar/', methods=['POST'])
def classifyZheimar_image():
    # load image from request
    imageZheimar = request.files['imageZehimar'].read()

    # preprocess image
    imageZheimar = preprocessZheimar_image(imageZheimar)

    # run inference on TensorFlow model
    resultZheimar = modelZheimar.predict(imageZheimar)

    # format the results
    labelsZheimar = ['MildDemented', 'ModerateDemented',
                     'NonDemented', 'VeryMildDemented']
    ResultZheimar = {}

    maxZheimar_prob = 0.0

    maxZheimar_label = ''
    for i, labelZheimar in enumerate(labelsZheimar):
        probZheimar = float(resultZheimar[0][i])
        ResultZheimar[labelZheimar] = probZheimar
        if probZheimar > maxZheimar_prob:
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
    n_imgZheimar_size = cv2.resize(
        imageZheimar, (224, 224), interpolation=cv2.INTER_LINEAR)

    return np.array([n_imgZheimar_size])


# end point Gastro 


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

    return jsonify(Result={'  Diagnose': maxGastro_label,  'accuracy': accGastro})

    # return "hello"




def preprocessGastro_image(imageGastro):
    # preprocess the imageGastro (resize, normalize, etc.)
    print(imageGastro)
    imageGastro = np.asarray(bytearray(imageGastro), dtype="uint8")
    imageGastro = cv2.imdecode(imageGastro, cv2.IMREAD_COLOR)

    # n_img = cv2.imread(imageGastro, cv2.IMREAD_COLOR)
    n_imgGastro_size = cv2.resize(
        imageGastro, (65, 65), interpolation=cv2.INTER_LINEAR)

    return np.array([n_imgGastro_size])





if __name__ == 'main':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000, host='0.0.0.0')

