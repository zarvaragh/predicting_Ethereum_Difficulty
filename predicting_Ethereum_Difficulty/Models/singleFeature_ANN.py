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
#plt.savefig("1- test_train.png", dpi = 1000, bbox_inches='tight')

sc = MinMaxScaler()
train_sc = sc.fit_transform(train)
test_sc = sc.fit_transform(test)



X_train = train_sc[:-1] 
y_train = train_sc[1:]

X_test = test_sc[:-1]
y_test = test_sc[1:]




from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model

K.clear_session()

model = Sequential()
#model.add(Dense(output_dim = 1, 4, input_dim=11, activation='relu'))
model.add(Dense(50, input_dim=1, activation='relu'))
#model.add(Dense(1))
model.add(Dense(units = 1, activation='linear'))
#model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])
model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=50, verbose=1)
history = model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1, callbacks=[early_stop], shuffle=False)

y_pred_test_ann = model.predict(X_test)

score_ann= model.evaluate(X_test, y_test, batch_size=1)
print('ANN: %f'%score_ann)
print('ANN: %f'%((1-score_ann)*100))

plt.figure(figsize = (9,5))
plt.plot(y_test, label='True')
plt.plot(y_pred_test_ann, label='ANN')
plt.title("ANN's Prediction")
plt.xlabel('Observations')
plt.ylabel('Scaled Difficulty Values')
plt.legend()
#plt.savefig("Pred-ANN_98.07_batch32.png", dpi = 1000, bbox_inches='tight')
plt.show()

plt.figure(figsize = (9,5))
plt.scatter(x = range(0, y_test.size) , y = y_test, color = 'b', marker='*', label = 'Actual')
plt.scatter(x = range(0, y_pred_test_ann.size) , y = y_pred_test_ann, color = 'r', marker='o', label = 'Predicted')
plt.title('Difficulty Prediction')
plt.xlabel('Timestamp')
plt.ylabel('Difficulty')
plt.legend(loc='best')
plt.savefig("pred-ann epoch_batch1.png", dpi = 1000, bbox_inches='tight')
plt.show()


