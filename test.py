import numpy as np
import tensorflow as tf
import os

from preprocess_images import imgs_to_dict, images_to_arr


dirpath = os.path.join(os.getcwd(), "integer_images")

d = imgs_to_dict(dirpath)
a = images_to_arr(d)
targets = np.array([int(i) for i in list(d.keys())])

x_train, x_test = a[:30], a[30:40]
y_train, y_test = np.array(targets[:30]), np.array(targets[30:40]) 
x_val, y_val = np.array(a[40:50]), np.array(targets[40:50])

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_val.shape)
print(y_val.shape)

data_augmentation = tf.keras.Sequential(
  [
    tf.keras.layers.RandomFlip("horizontal", input_shape=(44, 92, 1)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
  ]
)

def build_model(input_shape, n_outputs):
    model = tf.keras.Sequential([
        #data_augmentation,
        tf.keras.layers.Reshape(target_shape=(input_shape[0], input_shape[1], 1)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        #tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(n_outputs, activation="softmax")
    ])
    model.compile(optimizer='adam',
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
    return model

model = build_model((44, 92), 130)
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=60)
# model.evaluate(x_val, y_val)
scores = history.history
# loss = np.average(scores["loss"])
# accuracy = np.average(scores["accuracy"])
# print("Loss Average: {}".format(loss))
# print("Accuracy Average: {}".format(accuracy))
print("Loss: {}".format(scores["loss"][-1]))
print("Accuracy: {}".format(scores["accuracy"][-1]))