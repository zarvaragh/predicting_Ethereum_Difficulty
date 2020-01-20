import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('day.csv')
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index(["timestamp"], drop=True)
df = df.drop(['number','timestamp(days)', 'timestamp(time)', 'total_difficulty', 'col_no_after_duplication'], axis=1)

#Changin the order of the columns
df = df[['gas_limit', 'gas_used', 'size', 'transaction_count','difficulty']]

df.describe()
df.columns
df['difficulty'].plot()


split_date = pd.Timestamp('2018-08-01')

train = df.loc[:split_date]
test = df.loc[split_date:]

ax = train.plot(figsize = (12,7))
test.plot(ax=ax)
plt.legend(['train', 'test'])
#plt.savefig("1- test_train.png", dpi = 1000, bbox_inches='tight')

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
train = sc.fit_transform(train)
test = sc.fit_transform(test)

X_train = train[:, :-1] #all rows , all columns except the last one
y_train = train[:, -1] # all rows, only last column

X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


X_test = test[:, :-1]
y_test =  test[:, -1]

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

#layer 1
regressor.add(LSTM(units = 50,return_sequences = True, activation='relu', input_shape = (X_train.shape[1], 1)))
#regressor.add(Dropout(0.2))

#layer2
regressor.add(LSTM(units = 50, activation='relu'))
#regressor.add(Dropout(0.2))

#layer3
regressor.add(LSTM(units = 25, activation='relu', return_sequences = True))
#regressor.add(Dropout(0.2))

#Layer 4
regressor.add(LSTM(units = 25, activation='relu'))
#regressor.add(Dropout(0.2))

#output layer
regressor.add(Dense(units = 1, activation='linear')) # to make a full connection step 9 deep learning A-Z

#Compiling RNN
#adam=optimizers.Adam(lr=0.01, beta_1=0.91, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
#model.compile (loss ="mean_squared_error" , optimizer = "adam") 
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') #for continuous values we choose mean squared error


#Fitting RNN to Training set
history = regressor.fit(X_train, y_train, epochs = 100, batch_size = 64, validation_data=(X_test, y_test), shuffle=False )

#making predictions
# serialize model to JSON
model_json = regressor.to_json()
with open("model_LSTM.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
regressor.save_weights("model_LSTM.h5")
print("Saved model to disk")


y_pred = regressor.predict(X_test)

#inverse transform method for scaling back to original value
#pred = sc.inverse_transform(pred)

score_lstm= regressor.evaluate(X_test, y_test, batch_size=1)
print('LSTM: %f'%score_lstm)
print('LSTM: %f'%((1-score_lstm)*100))

# Line Graph Visulazation 
plt.figure(figsize = (9,5))
plt.plot(y_test,  label = 'Original Value')
plt.plot(y_pred,  label = 'Predicted Value')
plt.title('Difficulty Prediction')
plt.xlabel('Timestamp')
plt.ylabel('Difficulty')
plt.legend(loc='best')
#plt.savefig("3-LSTM_prediction_100 epoch_88.87_batch32.png", dpi = 1000, bbox_inches='tight')
plt.show()


#Scatter plot visualization
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
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
#plt.savefig("1-LSTM_loss_1000 epoch_88.87_batch256.png", dpi = 1000, bbox_inches='tight')
plt.show()



















#Both Graphs visualization
plt.figure(figsize = (9,5))
#plt.subtitle('owner', y = 0.94)

plt.subplot(1,1,1)
plt.plot(y_test,  label = 'Original Value')
plt.plot(y_pred,  label = 'Predicted Value')
plt.title('Difficulty Prediction')
plt.xlabel('Timestamp')
plt.ylabel('Difficulty')
plt.legend(loc='best')


plt.subplot(1,2,1)
plt.figure(figsize = (9,5))
plt.scatter(x = range(0, y_test.size) , y = y_test, color = 'b', marker='*', label = 'Actual')
plt.scatter(x = range(0, y_pred.size) , y = y_pred, color = 'r', marker='o', label = 'Predicted')
plt.title('Difficulty Prediction')
plt.xlabel('Timestamp')
plt.ylabel('Difficulty')
plt.legend(loc='best')
#plt.show()




#Evaluating
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(y_test, y_pred))











