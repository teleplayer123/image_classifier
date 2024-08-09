import numpy as np
import tensorflow as tf

from preprocess_images import imgs_to_dict, images_to_arr



d = imgs_to_dict("integer_images")
a = images_to_arr(d)
print(a.shape)

orig_targets = ["89", "121", "28", "86", "121", "33", "81", "119", "28", "62", "120", "32", "75", "121", "79", "80", "121", "52", "76", "120", "39", "81", "120", "45", "81", "120", "45", "72", "121", "30", "69", "123", "57", "62", "120", "32", "81", "120", "53", "87"]
targets = [int(i) for i in orig_targets]

x_train, x_test = a[:30], a[30:40]
y_train, y_test = np.array(targets[:30]), np.array(targets[30:40]) 

print(x_train.shape)
print(y_train.shape)

def build_model(input_shape, n_outputs):
    model = tf.keras.Sequential([
        # tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Reshape(target_shape=(input_shape[0], input_shape[1], 1)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        # tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(n_outputs, activation="softmax")
    ])
    model.compile(optimizer='adam',
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
    return model


model = build_model((44, 92), 125)
model.fit(x_train, y_train, epochs=40)
scores = model.evaluate(x_test, y_test, verbose=0)
print(f"Loss: {scores[0]}")
print(f"Accuracy: {scores[1]}")

