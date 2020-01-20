#https://github.com/neelabhpant/Deep-Learning-in-Python/blob/master/usd_to_inr_models.ipynb

import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from flask import request, json


def adj_r2_score(r2, n, k):
    return 1-((1-r2)*((n-1)/(n-k-1)))

df = pd.read_csv('day.csv')
df = df[[ 'gas_limit', 'gas_used', 'size', 'timestamp','transaction_count', 'difficulty']]
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

split_date = pd.Timestamp('2018-05-01')

train = df.loc[:split_date]
test = df.loc[split_date:]

ax = train['difficulty'].plot(figsize = (12,7))
test['difficulty'].plot(ax=ax)
plt.legend(['train', 'test'])
#plt.savefig("1- test_train.png", dpi = 1000, bbox_inches='tight')

sc = MinMaxScaler()
train_sc = sc.fit_transform(train)
test_sc = sc.fit_transform(test)


X_train = train_sc[:, :-1]
y_train = train_sc[:, -1]

X_test = test_sc[:, :-1]
y_test = test_sc[:, -1]

#ANN 
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model

K.clear_session()

model = Sequential()
#model.add(Dense(output_dim = 1, 4, input_dim=11, activation='relu'))
model.add(Dense(50, input_dim=4, activation='relu'))
#model.add(Dense(70, input_dim=4, activation='relu'))
model.add(Dense(1, activation='linear'))
#model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])
model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=50, verbose=1)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, validation_data=(X_test, y_test), callbacks=[early_stop], shuffle=False)


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")



#model.save('ANN_multiple_new.h5')
#my_ann_model = load_model("ANN_multiple_new.h5")



y_train_pred_ann = model.predict(X_train)
y_pred_test_ann = model.predict(X_test)

y_pred_test_ANN = model.predict(X_test)
score_ann= model.evaluate(X_test, y_test)
print((1-score_ann)*100)




plt.figure(figsize = (9,5))
plt.plot(y_test, label='True')
plt.plot(y_pred_test_ANN, label='Predicted')
plt.title("ANN's_Prediction")
plt.xlabel('Observation')
plt.ylabel('INR_Scaled')
plt.legend()
#plt.savefig("2- ANN_prediction_100_93.22_batch32.png", dpi = 1000, bbox_inches='tight')
plt.show()


plt.figure(figsize = (9,5))
plt.scatter(x = range(0, y_test.size) , y = y_test, color = 'b', marker='*', label = 'Actual')
plt.scatter(x = range(0, y_pred.size) , y = y_pred, color = 'r', marker='o', label = 'Predicted')
plt.title('Difficulty Prediction')
plt.xlabel('Timestamp')
plt.ylabel('Difficulty')
plt.legend(loc='best')
#plt.savefig("4-LSTM_prediction_1000 epoch_batch512.png", dpi = 1000, bbox_inches='tight')
plt.show()



# summarize history for loss
plt.figure(figsize = (9,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('ANN Loss Trend')
plt.ylabel('Loss Values')
plt.xlabel('Number of Epochs')
plt.legend(['train', 'test'], loc='upper right')
#plt.savefig("1- ANN_loss_100_93.22_batch32.png", dpi = 1000, bbox_inches='tight')
plt.show()



















#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred_test_ann)
#
#
#import numpy as np
#from flask import request, Flask, jsonify, abort
#
#
#my_ann_model = load_model('ANN_multiple_new.h5')
#
#app = Flask(__name__)
#
#app.route('/api', methods=['POST'])
#def make_predict():
#    data = request.get_json(force=True)
#    predict_request = [data['gas_limit'], data['gas_used'], data['size'], data['transaction_count']]
#    predict_request = np.array(predict_request)
#    y_hat = my_ann_model.predict(predict_request)
#    output = [y_hat[0]]
#    return jsonify(result=output)
#
##if __name__ == '__main__':
#app.run(port=5000, debug=True)
#




