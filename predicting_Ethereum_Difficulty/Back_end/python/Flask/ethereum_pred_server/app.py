import tensorflow as tf
tf.__version__
import pandas as pd



from flask import Flask, request, jsonify, abort, render_template
import numpy as np
from keras.models import load_model
from keras.models import model_from_json
import tensorflow as tf
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# ANN model importing ...
json_file_ANN = open('model_ANN.json', 'r')
loaded_model_json_ANN = json_file_ANN.read()
json_file_ANN.close()
loaded_model_ANN = model_from_json(loaded_model_json_ANN)
loaded_model_ANN.load_weights("model_ANN.h5")
print("Loaded ANN model from disk")
graph_ANN = tf.get_default_graph()
# =================================================================

# LSTM model importing ...
json_file_LSTM = open('model_LSTM.json', 'r')
loaded_model_json_LSTM = json_file_LSTM.read()
json_file_LSTM.close()
loaded_model_LSTM = model_from_json(loaded_model_json_LSTM)
loaded_model_LSTM.load_weights("model_LSTM.h5")
print("Loaded LSTM model from disk")
graph_LSTM = tf.get_default_graph()
# =================================================================


@app.route('/home')
def index():
    return render_template('index.html')
#my_ann_model = load_model('ANN_multiple_new.h5')


@app.route('/predict', methods=['GET', 'POST'])
def pred():
    if (request.method == 'GET'):
        params = request.args
    if (request.method == 'POST'):
        params = request.json

    if (params != None):
        gas_limit = params.get('gas_limit')
        gas_used = params.get('gas_used')
        size = params.get('size')
        transaction_count = params.get('transaction_count')
        # date = params.get('date')
        gas_limit = np.float64(gas_limit)
        gas_used = np.float64(gas_used)
        size = np.float64(size)
        transaction_count = np.float64(transaction_count) 

# ANN Prediction
    with graph_ANN.as_default():
        result_ANN = loaded_model_ANN.predict(
            np.array([[gas_limit, gas_used, size, transaction_count]]))[0].tolist()
        print('ANN Prediction: ')
        print(result_ANN)
        print('%'*40)

# LSTM Prediction
    with graph_LSTM.as_default():
        result_LSTM = loaded_model_LSTM.predict(
            np.array([[[gas_limit], [gas_used], [size], [transaction_count]]]))[0].tolist()
        print('LSTM Prediction: ')
        print(result_LSTM)
        print('%'*40)

    data = {
        'ann_prediction': result_ANN,
        'lstm_prediction': result_LSTM
    }
    print('='*40)
    return (jsonify(data))


app.run(port=5000)
