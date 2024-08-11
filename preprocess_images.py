import cv2
import os
import numpy as np


########  NOTES  ##########

# Tensor Info:
#  Shape: length of each of the axes of the tensor
#  Rank: Number of tensor axes.
#        - Scalar has rank 0
#        - Vector has rank 1
#        - Matrix has rank 2
# Axis/Dimension: a particular dimension of a tensor
# Size: total number of items in the tensor, the product of the shape vector's elements

# Typical axis order (Rank 4): (Batch, Height, Width, Features)


##################################################################
#       Functions that were used to create dataset               #
##################################################################

def split_image_digits(path):
    img = cv2.imread(path)
    img1 = img[0:44, 0:92]
    img2 = img[0:44, 92:184]
    img3 = img[0:44, 184:276]
    # img1 = img1.reshape(-1, 4048).astype(np.int8)
    # img2 = img2.reshape(-1, 4048).astype(np.int8)
    # img3 = img3.reshape(-1, 4048).astype(np.int8)
    return img1, img2, img3

def crop_images(paths):
    i = 0
    img_dir = os.path.join(os.getcwd(), "datasets", "images")
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    for path in paths:
        i += 1
        new_img_path = os.path.join(img_dir, "img{}.png".format(i))
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = img_data[0:45, 0:278]
        cv2.imwrite(new_img_path, img)

def create_image_data(img_dirname, outdirname="newer_images"):
    images_dir = os.path.join(os.getcwd(), "datasets", img_dirname)
    outdir = os.path.join(os.getcwd(), "datasets", outdirname)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    paths = [[os.path.join(dn, fn) for fn in files] for dn, _ , files in os.walk(images_dir)]
    paths = sum(paths, [])
    crop_images(paths)
    i = 0
    for path in paths:
        img1, img2, img3 = split_image_digits(path)
        path1 = os.path.join(outdir, "image{}.png".format(i+1))
        path2 = os.path.join(outdir, "image{}.png".format(i+2))
        path3 = os.path.join(outdir, "image{}.png".format(i+3))
        cv2.imwrite(path1, img1)
        cv2.imwrite(path2, img2)
        cv2.imwrite(path3, img3)
        i += 3
    return outdir

#######################################################
#                Preprocessing Functions              #
#######################################################

def normalize_img(img):
    img = img * 1.0/255
    return img

def imgs_to_dict(image_dir):
    img_dict = {}
    for fname in os.listdir(image_dir):
        img_path = os.path.join(image_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        name = fname.split(".")[0]
        idx = name.split("image")[-1]
        img_dict[idx] = img
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
