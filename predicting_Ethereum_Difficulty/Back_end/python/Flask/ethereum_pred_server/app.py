import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# tf.get_default_graph() was removed in TF2; models are loaded directly
loaded_model_ANN = keras.models.model_from_json(
    open("model_ANN.json").read()
)
loaded_model_ANN.load_weights("model_ANN.h5")
print("Loaded ANN model from disk")

loaded_model_LSTM = keras.models.model_from_json(
    open("model_LSTM.json").read()
)
loaded_model_LSTM.load_weights("model_LSTM.h5")
print("Loaded LSTM model from disk")


@app.route("/home")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def pred():
    params = request.args if request.method == "GET" else request.json
    if not params:
        return jsonify({"error": "No parameters provided"}), 400

    try:
        gas_limit = np.float64(params["gas_limit"])
        gas_used = np.float64(params["gas_used"])
        size = np.float64(params["size"])
        transaction_count = np.float64(params["transaction_count"])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    result_ann = loaded_model_ANN.predict(
        np.array([[gas_limit, gas_used, size, transaction_count]])
    )[0].tolist()

    result_lstm = loaded_model_LSTM.predict(
        np.array([[[gas_limit], [gas_used], [size], [transaction_count]]])
    )[0].tolist()

    return jsonify({"ann_prediction": result_ann, "lstm_prediction": result_lstm})


if __name__ == "__main__":
    app.run(port=5000)