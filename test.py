from crop_images import save_image_data, create_image_data
import os

paths = create_image_data("images")
imgs_path = os.path.join(os.getcwd(), "datasets", "new_images")
df = save_image_data(imgs_path)
print(df)