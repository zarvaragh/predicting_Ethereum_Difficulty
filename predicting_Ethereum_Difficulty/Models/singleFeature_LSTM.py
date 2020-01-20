import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

def adj_r2_score(r2, n, k):
    return 1-((1-r2)*((n-1)/(n-k-1)))

df = pd.read_csv('day.csv')
df["timestamp"] = pd.to_datetime(df["timestamp"])
ind_df = df.set_index(["timestamp"], drop=True)

data_frame = ind_df.sort_index(axis=1 ,ascending=True)

df_single = data_frame[["difficulty"]]
df_single.plot()

split_date = pd.Timestamp('2018-05-01')

train = df_single.loc[:split_date]
test = df_single.loc[split_date:]

ax = train['difficulty'].plot(figsize = (12,7))
test['difficulty'].plot(ax=ax)
plt.legend(['train', 'test'])


sc = MinMaxScaler()
train_sc = sc.fit_transform(train)
test_sc = sc.fit_transform(test)

X_train = train_sc[:-1] 
y_train = train_sc[1:]

X_test = test_sc[:-1]
y_test = test_sc[1:]

X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#X_tr_t = X_train.reshape(X_train.shape[0], 1, X_train.shape[1]) # X_train.shape[0] exactly will give u the number of observation, so either can write the whole number or type it like this so that the code can be reused ,, X_train.shape[1] will give the number of columns and 1 is the number of indicators that we wanna add. 
#X_tst_t = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))



from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model

#Clearing the sessions and memory 
K.clear_session()

#initializing the model
model_lstm = Sequential()

#First hidden layer
model_lstm.add(LSTM(50, input_shape=(1, X_train.shape[1]), activation='relu', kernel_initializer='lecun_uniform', return_sequences=True))

#Second hidden layer
model_lstm.add(LSTM(units = 50, activation='relu'))

#output layer
model_lstm.add(Dense(units = 1, activation='linear'))

#model_lstm.add(Dense(1))
#model_lstm.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])

model_lstm.compile(loss='mean_squared_error', optimizer='adam')

early_stop = EarlyStopping(monitor='loss', patience=50, verbose=1)

history_model_lstm = model_lstm.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, shuffle=False, callbacks=[early_stop],  validation_data=(X_test, y_test))

y_pred_test_lstm = model_lstm.predict(X_test)

score_lstm= model_lstm.evaluate(X_test, y_test, batch_size=1)
print('LSTM: %f'%score_lstm)
print('LSTM: %f'%((1-score_lstm)*100))

plt.figure(figsize = (9,5))
plt.plot(y_test, label='True')
plt.plot(y_pred_test_lstm, label='LSTM')
plt.title("LSTM's Prediction")
plt.xlabel('Observations')
plt.ylabel('Scaled Difficulty Values')
plt.legend()
#plt.savefig("pred-LSTM_single_32batch_96.69.png", dpi = 1000, bbox_inches='tight')
plt.show()

plt.figure(figsize = (9,5))
plt.scatter(x = range(0, y_test.size) , y = y_test, color = 'b', marker='*', label = 'Actual')
plt.scatter(x = range(0, y_pred_test_lstm.size) , y = y_pred_test_lstm, color = 'r', marker='o', label = 'Predicted')
plt.title('Difficulty Prediction')
plt.xlabel('Timestamp')
plt.ylabel('Difficulty')
plt.legend(loc='best')
plt.savefig("pred-lstm_ epoch_batch32.png", dpi = 1000, bbox_inches='tight')
plt.show()
