import numpy as np
import tensorflow as tf
import os


data_dir = os.path.join(os.getcwd(), "digit_images_dataset")
# data_dir = os.path.join(os.getcwd(), "integer_dataset")

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(92, 92),
  batch_size=32)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(92, 92),
  batch_size=32)


def build_model(input_shape, n_outputs):
   model = tf.keras.models.Sequential()
   model.add(tf.keras.layers.Input((input_shape[0], input_shape[1], input_shape[2])))
   model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
   model.add(tf.keras.layers.MaxPool2D((2, 2)))
   model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
   model.add(tf.keras.layers.MaxPool2D((2, 2)))
   model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
   model.add(tf.keras.layers.Flatten())
   model.add(tf.keras.layers.Dense(128, activation="relu"))
   model.add(tf.keras.layers.Dense(n_outputs))
   model.compile(optimizer="adam",
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=["accuracy"])
   return model

def convert_model_to_tflite(model):
  file_path = os.path.join(os.getcwd(), "models", "model.tflite")
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  with open(file_path, "wb") as fh:
    fh.write(tflite_model)

def save_tf_model(model):
  model_dir = os.path.join(os.getcwd(), "models")
  if not os.path.exists(model_dir):
      os.mkdir(model_dir)
  save_dir = os.path.join(model_dir, "saved_models")
  if not os.path.exists(save_dir):
      os.mkdir(save_dir)
  tf.saved_model.save(model, save_dir)

def convert_tflite_int8(saved_model_dir, input_shape=(92, 92, 3), n_outputs=10):
  def representative_dataset():
    for _ in range(n_outputs):
      data = np.random.rand(1, input_shape[0], input_shape[1], input_shape[2])
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

labels_path = os.path.join(os.getcwd(), "models", "labels.txt")
class_names = train_ds.class_names
AUTOTUNE = tf.data.AUTOTUNE

labels = np.array(class_names)
labels.tofile(labels_path, sep="\n")


train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))

model = build_model((92, 92, 3), 10)
history = model.fit(train_ds, validation_data=val_ds, epochs=20)

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

print("Score Averages\n--------------------")
print(f"Accuracy: {np.average(acc)}")
print(f"Loss: {np.average(loss)}")
print(f"Val Accuracy: {np.average(val_acc)}")
print(f"Val Loss: {np.average(val_loss)}")
print("\nScores\n-------------------------")
print(f"Accuracy: {acc[-1]}")
print(f"Loss: {loss[-1]}")
print(f"Val Accuracy: {val_acc[-1]}")
print(f"Val Loss: {val_loss[-1]}")

def predict_digit(digit_path):
  img = tf.keras.utils.load_img(digit_path, target_size=(92, 92))
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0)
  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])
  print("This image most likely belongs to {} with a {:.2f} percent confidence".format(class_names[np.argmax(score)], 100 * np.max(score)))


saved_model_dir = os.path.join(os.getcwd(), "models", "saved_models")

save_tf_model(model)
convert_model_to_tflite(model)
convert_tflite_int8(saved_model_dir)
#digit_path = os.path.join(os.getcwd(), "digit_images_dataset", "5_five", "image5_0.png")
#predict_digit()