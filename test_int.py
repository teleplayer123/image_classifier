from pycoral.utils.edgetpu import make_interpreter
from image_classification import classify_image
import os

labels = os.path.join(os.getcwd(), "models_int", "labels.txt")
input_img = os.path.join(os.getcwd(), "integer_image_dataset", "image89_0.png")
model_file = os.path.join(os.getcwd(), "models_int", "model.tflite")

interpreter = make_interpreter(model_file)

classify_image(input_img, labels, interpreter)