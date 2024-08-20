from pycoral.utils.edgetpu import make_interpreter
from image_classification import classify_image
import os

labels = os.path.join(os.getcwd(), "models", "labels.txt")
# input_img = os.path.join(os.getcwd(), "digit_images_dataset", "9_nine", "image9_0.png")
# input_img = os.path.join(os.getcwd(), "integer_dataset", "32", "image32_0.png")
model_file = os.path.join(os.getcwd(), "models", "model.tflite")

interpreter = make_interpreter(model_file)

for i in range(20):
    try:
        input_img = os.path.join(os.getcwd(), "integer_dataset", "{}".format(i), "image{}_0.png".format(i))
        classify_image(input_img, labels, interpreter)
        print("Actual value: {}".format(i))
    except TypeError:
        continue