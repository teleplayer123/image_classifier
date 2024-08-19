import os
import tensorflow as tf
import numpy as np
import cv2


#####################################
#       Convert/Save Functions      #
#####################################

def convert_saved_model_to_tflite(saved_model_dir):
  file_path = os.path.join(os.getcwd(), "models", "model.tflite")
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
  tflite_model = converter.convert()
  with open(file_path, "wb") as fh:
    fh.write(tflite_model)

def convert_model_to_tflite(model, outdir="models"):
  file_path = os.path.join(os.getcwd(), outdir, "model.tflite")
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  with open(file_path, "wb") as fh:
    fh.write(tflite_model)

def save_tf_model(model, outdir="models"):
  model_dir = os.path.join(os.getcwd(), outdir)
  if not os.path.exists(model_dir):
      os.mkdir(model_dir)
  save_dir = os.path.join(model_dir, "saved_models")
  if not os.path.exists(save_dir):
      os.mkdir(save_dir)
  tf.saved_model.save(model, save_dir)

def convert_tflite_int8(saved_model_dir, input_shape=(92, 92, 3), n_outputs=130, outdir="models"):
  def representative_dataset():
    for _ in range(n_outputs):
      data = np.random.rand(1, input_shape[0], input_shape[1], input_shape[2])
      yield [data.astype(np.float32)]
      
  file_path = os.path.join(os.getcwd(), outdir, "model.tflite")
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.representative_dataset = representative_dataset
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.inference_input_type = tf.uint8
  converter.inference_output_type = tf.uint8
  tflite_quant_model = converter.convert()
  with open(file_path, "wb") as fh:
    fh.write(tflite_quant_model)


#####################################
#         Model Functions           #
#####################################

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

def build_model_v2(input_shape, n_outputs):
   model = tf.keras.models.Sequential()
   model.add(tf.keras.layers.Input((input_shape[0], input_shape[1], 1)))
   model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
   model.add(tf.keras.layers.MaxPool2D((2, 2)))
   model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
   model.add(tf.keras.layers.MaxPool2D((2, 2)))
   model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
   model.add(tf.keras.layers.Flatten())
   model.add(tf.keras.layers.Dense(64, activation="relu"))
   model.add(tf.keras.layers.Dense(n_outputs, activation="softmax"))
   model.compile(optimizer="adam",
                 loss="sparse_categorical_crossentropy",
                 metrics=["accuracy"])
   return model

def build_model_aug(input_shape, n_outputs):
    data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal", input_shape=(input_shape[0], input_shape[1])),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    model = tf.keras.Sequential([
        data_augmentation,
        tf.keras.layers.Reshape(target_shape=(input_shape[0], input_shape[1], 1)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(n_outputs, activation="softmax")
    ])
    model.compile(optimizer='adam',
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
    return model

def train_model_avg(X, y, epochs=100):
    x_train, x_test = X[:30], X[30:40]
    y_train, y_test = np.array(y[:30]), np.array(y[30:40]) 
    x_val, y_val = np.array(X[40:50]), np.array(y[40:50])
    model = build_model((44, 92), 130)
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs)
    # model.evaluate(x_val, y_val)
    scores = history.history
    loss = np.average(scores["loss"])
    accuracy = np.average(scores["accuracy"])
    print("Loss Average: {}".format(loss))
    print("Accuracy Average: {}".format(accuracy))
    print("Loss: {}".format(scores["loss"][-1]))
    print("Accuracy: {}".format(scores["accuracy"][-1]))


#################################
#      Extra Model Builds       # 
#################################

def build_model_basic(input_shape, n_outputs):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(n_outputs, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

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
    model = tf.keras.models.Sequential()
    for i in range(n_layers):
        if i == 0:
            model.add(tf.keras.layers.LSTM(n_nodes, return_sequences=True, input_shape=input_shape))
        elif i == n_layers-1:
            model.add(tf.keras.layers.LSTM(n_nodes))
        else:
            model.add(tf.keras.layers.LSTM(n_nodes, return_sequences=True))
        model.add(tf.keras.layers.Dropout(drop_rate))
        if batch_normalization:
            model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(n_out, activation=activation))
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model


###################################################
#            Preprocessing Functions              #
###################################################

def normalize_img(img):
    img = img * 1.0/255
    return img

def imgs_to_dict(image_dir):
    img_dict = {}
    for fname in os.listdir(image_dir):
        img_path = os.path.join(image_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        names = fname.split("_")
        name = names[0]
        sub_idx = names[-1].split(".")[0]
        idx = name.split("image")[-1]
        img_dict[str(idx)+"_"+str(sub_idx)] = img
    return img_dict

def images_to_arr(obj):
    imgs = []
    if isinstance(obj, dict):
        imgs = [normalize_img(img) for img in obj.values()]
    elif isinstance(obj, str):
        dir_path = None 
        if os.path.isdir(imgs):
            dir_path = imgs
        else:
            dir_path = os.path.join(os.getcwd(), obj)
        for fname in sorted(os.listdir(dir_path)):
            img_path = os.path.join(dir_path, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = normalize_img(img)
            imgs.append(img)
    else:
        raise TypeError(f"type {type(obj)} is not supported")
    return np.array(imgs)

def imgs_to_dict_v1(image_dir):
    img_dict = {}
    for fname in os.listdir(image_dir):
        img_path = os.path.join(image_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        name = fname.split(".")[0]
        idx = name.split("image")[-1]
        img_dict[idx] = img
    return img_dict

def images_to_arr_v1(obj):
    imgs = []
    if isinstance(obj, dict):
        imgs = [normalize_img(img) for img in obj.values()]
    elif isinstance(obj, str):
        dir_path = None 
        if os.path.isdir(imgs):
            dir_path = imgs
        else:
            dir_path = os.path.join(os.getcwd(), obj)
        for fname in sorted(os.listdir(dir_path)):
            img_path = os.path.join(dir_path, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = normalize_img(img)
            imgs.append(img)
    else:
        raise TypeError(f"type {type(obj)} is not supported")
    return np.array(imgs)

def zero_pad(img, shape=(92, 92)):
    img_shape = np.shape(img)
    if img_shape[0] > shape[0] or img_shape[1] > shape[1]:
        raise ValueError("Shape to pad image to must be greater or equal to image original shape.")
    x_size = shape[0] - img_shape[0]
    x1 = x_size // 2
    x2 = x_size // 2
    xs = sum([img_shape[0], x1, x2])
    assert xs == shape[0], "Not able to pad image into provided shape"
    y_size = shape[1] - img_shape[1]
    y1 = y_size // 2
    y2 = y_size // 2
    ys = sum([img_shape[1], y1, y2])
    assert ys == shape[1], "Not able to pad image into provided shape"
    padded_img = np.pad(img, ((x1, x2), (y1, y2)), mode="constant", constant_values=0)
    return padded_img