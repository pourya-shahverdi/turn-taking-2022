import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
model.add(layers.LSTM(30, kernel_regularizer=keras.regularizers.L2(0.001)))
model.add(layers.Dense(60, activation='sigmoid', kernel_regularizer=keras.regularizers.L2(0.01)))

model.compile(loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.MeanAbsoluteError()])