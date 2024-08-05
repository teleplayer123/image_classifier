from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
import tensorflow as tf


##################################################################
#                main models for use                             #
##################################################################


def build_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(4048,)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(1000, activation='softmax'))
    #model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(X, y, epochs=100):
    X_train, X_test, y_train, y_test = X[:20], X[20:25], y[:20], y[20:25]
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)
    model = build_model()
    res = model.fit(X_train, y_train, epochs=epochs)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f"Loss: {scores[0]}")
    print(f"Accuracy: {scores[1]}")


##################################################################
#            other models for reference                          #
##################################################################


def build_scn(num_classes=1000):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer='adam', loss=tf.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model


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