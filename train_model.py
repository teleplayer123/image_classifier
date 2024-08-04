from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
import tensorflow as tf


def create_model():
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(19,)),
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(3)
    ])



def build_n_layer_model(n_nodes, n_layers, input_shape, n_out=2, drop_rate=0.2, batch_normalization=True,
                        loss="binary_crossentropy", opt="adam", metrics=["accuracy"],
                        activation="softmax"):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            model.add(LSTM(n_nodes, return_sequences=True, input_shape=input_shape))
        elif i == n_layers-1:
            model.add(LSTM(n_nodes))
        else:
            model.add(LSTM(n_nodes, return_sequences=True))
        model.add(Dropout(drop_rate))
        if batch_normalization:
            model.add(tf.keras.layers.BatchNormalization())
    model.add(Dense(n_out, activation=activation))
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model