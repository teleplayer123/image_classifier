from pycoral.utils.edgetpu import make_interpreter
from image_classification import classify_image
import os

labels = os.path.join(os.getcwd(), "models", "labels.txt")
# input_img = os.path.join(os.getcwd(), "digit_images_dataset", "9_nine", "image9_0.png")
input_img = os.path.join(os.getcwd(), "integer_image_dataset", "32", "image32_0.png")
model_file = os.path.join(os.getcwd(), "models", "model.tflite")

interpreter = make_interpreter(model_file)

classify_image(input_img, labels, interpreter)