import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from matplotlib import pyplot as plt
from tensorflow import keras
import matplotlib.image as mpimg


app = Flask(__name__)

model = keras.models.load_model('quality_control.h5')

def replace_backslash(string):
    return string.replace(' \\', '/')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    image_shape = (300, 300, 1)
    chemin = replace_backslash(request.form.get('image'))
    img_pred_default = cv2.imread(chemin)
    img_pred = cv2.imread(chemin, cv2.IMREAD_GRAYSCALE)
    img_pred = img_pred / 255  # rescale
    prediction =model.predict(img_pred.reshape(1, *image_shape), verbose=0)
    if (prediction < 0.5):
        predicted_label = "Ce produit présente des defauts à"
        prob = (1 - prediction.sum()) * 100

    else:
        predicted_label = "Produit Ok à"
        prob = prediction.sum() * 100


    # Chemin de l'image à sauvegarder
    image_path =chemin

    # Charger l'image
    img = mpimg.imread(image_path)

    # Chemin de sauvegarde de l'image
    save_path = "static/css/image.jpeg"

    # Enregistrer l'image
    mpimg.imsave(save_path, img)
    return render_template('index.html', prediction_text=predicted_label+" : {}".format(round(prob, 4))+"%",chemin=save_path)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)