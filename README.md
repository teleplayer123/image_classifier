# Image Classifiation 
Image classification for pictures of integers trained and converted for use with Coral TPU's leveraging the pycoral python module. 

# Notes
This change needs to be made in pycoral's common.py modules
to work with the example images the model was trained on:

```
#changes to lines 55-58 in pycoral/adapters/common.py

def input_size(interpreter):
  """Gets a model's input size as (width, height) tuple.

  Args:
    interpreter: The ``tf.lite.Interpreter`` holding the model.
  Returns:
    The input tensor size as (width, height) tuple.
  """
  try:
    _, height, width, _ = input_details(interpreter, 'shape')
  except ValueError:
    _, height, width = input_details(interpreter, 'shape')
  return width, height
```

And one more change here:
```
def set_input(interpreter, data):
  """Copies data to a model's input tensor.

  Args:
    interpreter: The ``tf.lite.Interpreter`` to update.
    data: The input tensor.
  """
  try:
    input_tensor(interpreter)[:, :] = data
  except ValueError:
    input_tensor(interpreter)[:, :, -1] = data
```