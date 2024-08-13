import numpy as np
import tensorflow as tf
import os

from utils import build_model, imgs_to_dict, images_to_arr


dirpath = os.path.join(os.getcwd(), "integer_image_dataset")

model_dir = os.path.join(os.getcwd(), "models")
if not os.path.exists(model_dir):
   os.mkdir(model_dir)

d = imgs_to_dict(dirpath)
a = images_to_arr(d)
targets = np.array([int(i.split("_")[0]) for i in list(d.keys())])

labels = np.sort(targets, axis=0)
label_file = os.path.join(os.getcwd(), "models", "labels.txt")
labels = list(set(labels))
with open(label_file, "w") as fh:
    for label in labels:
        fh.write(str(label))
        if label != labels[-1]:
           fh.write("\n")

x_train, x_test = a[:40], a[40:50]
y_train, y_test = np.array(targets[:40]), np.array(targets[40:50]) 
x_val, y_val = np.array(a[50:51]), np.array(targets[50:51])

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_val.shape)
print(y_val.shape)

model = build_model((92, 92), 130)
# model = build_model((38, 38), 130)
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=60)
model.evaluate(x_test, y_test)
scores = history.history
loss = np.average(scores["loss"])
accuracy = np.average(scores["accuracy"])
print("Loss Average: {}".format(loss))
print("Accuracy Average: {}".format(accuracy))
print("Loss: {}".format(scores["loss"][-1]))
print("Accuracy: {}".format(scores["accuracy"][-1]))


def save_tf_model(model):
  save_dir = os.path.join(os.getcwd(), "models", "saved_models")

  if not os.path.exists(save_dir):
      os.mkdir(save_dir)

  tf.saved_model.save(model, save_dir)

save_tf_model(model)

saved_model_dir = os.path.join(os.getcwd(), "models", "saved_models")

def convert_model_to_tflite(model):
  file_path = os.path.join(os.getcwd(), "models", "model.tflite")
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  with open(file_path, "wb") as fh:
    fh.write(tflite_model)

def convert_tflite_int8(saved_model_dir):
  def representative_dataset():
    for _ in range(130):
      data = np.random.rand(1, 92, 92, 1)
      # data = np.random.rand(1, 38, 38, 1)
      yield [data.astype(np.float32)]
      
  file_path = os.path.join(os.getcwd(), "models", "model.tflite")
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.representative_dataset = representative_dataset
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.inference_input_type = tf.uint8
  converter.inference_output_type = tf.uint8
  tflite_quant_model = converter.convert()
  with open(file_path, "wb") as fh:
    fh.write(tflite_quant_model)

convert_model_to_tflite(model)
convert_tflite_int8(saved_model_dir)

