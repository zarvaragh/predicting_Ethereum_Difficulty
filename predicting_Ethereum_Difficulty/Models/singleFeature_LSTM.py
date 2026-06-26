import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras

df = pd.read_csv("day.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp").sort_index()

df_single = df[["difficulty"]]

split_date = pd.Timestamp("2018-05-01")
train = df_single.loc[:split_date]
test = df_single.loc[split_date:]

sc = MinMaxScaler()
train_sc = sc.fit_transform(train)
test_sc = sc.fit_transform(test)

X_train = np.reshape(train_sc[:-1], (-1, 1, 1))
y_train = train_sc[1:]
X_test = np.reshape(test_sc[:-1], (-1, 1, 1))
y_test = test_sc[1:]

keras.backend.clear_session()

model = keras.Sequential([
    keras.layers.LSTM(50, input_shape=(1, 1), activation="relu",
                      kernel_initializer="lecun_uniform", return_sequences=True),
    keras.layers.LSTM(50, activation="relu"),
    keras.layers.Dense(1, activation="linear"),
])
model.compile(loss="mean_squared_error", optimizer="adam")
early_stop = keras.callbacks.EarlyStopping(monitor="loss", patience=50, verbose=1)
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1,
          shuffle=False, callbacks=[early_stop], validation_data=(X_test, y_test))

y_pred = model.predict(X_test)
score = model.evaluate(X_test, y_test, batch_size=1)
print(f"LSTM MSE: {score:.6f}")

model.save("model_LSTM.keras")

plt.figure(figsize=(9, 5))
plt.plot(y_test, label="True")
plt.plot(y_pred, label="LSTM")
plt.title("LSTM Prediction")
plt.xlabel("Observations")
plt.ylabel("Scaled Difficulty")
plt.legend()
plt.tight_layout()
plt.show()