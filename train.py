import numpy as np
import tensorflow as tf
import os

from utils import build_model, imgs_to_dict, images_to_arr, convert_tflite_int8, save_tf_model, convert_model_to_tflite


dirpath = os.path.join(os.getcwd(), "integer_image_dataset")

model_dir = os.path.join(os.getcwd(), "models")
if not os.path.exists(model_dir):
   os.mkdir(model_dir)

d = imgs_to_dict(dirpath)
a = images_to_arr(d)
targets = np.array([int(i.split("_")[0]) for i in list(d.keys())])

labels = targets
label_file = os.path.join(os.getcwd(), "models", "labels.txt")
labels.tofile(label_file, sep="\n", format="%d")

x_train, x_test = a[:100], a[100:117]
y_train, y_test = targets[:100], targets[100:117]

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

model = build_model((92, 92), 130)
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=60)
model.evaluate(x_test, y_test)
scores = history.history
print("Loss: {}".format(scores["loss"][-1]))
print("Accuracy: {}".format(scores["accuracy"][-1]))

saved_model_dir = os.path.join(os.getcwd(), "models", "saved_models")

save_tf_model(model)
convert_model_to_tflite(model)
convert_tflite_int8(saved_model_dir, input_shape=(92, 92), n_outputs=130)