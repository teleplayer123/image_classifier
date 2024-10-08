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


########################################################
#       Functions used to create dataset               #
########################################################

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

def zero_pad_img(img):
    if np.shape(img) == (32, 16):
        padded_img = np.pad(img, ((30, 30), (38, 38)), mode="constant", constant_values=0)
    elif np.shape(img) == (44, 92):
        padded_img = np.pad(img, ((24, 24), (0, 0)), mode='constant', constant_values=0)
    elif np.shape(img) == (32, 10):
        padded_img = np.pad(img, ((30, 30), (41, 41)), mode='constant', constant_values=0)
    elif np.shape(img) == (32, 18):
        padded_img = np.pad(img, ((30, 30), (37, 37)), mode='constant', constant_values=0)
    elif np.shape(img) == (32, 12):
        padded_img = np.pad(img, ((30, 30), (40, 40)), mode='constant', constant_values=0)
    else:
        raise ValueError("image has unexpexted shape: {}".format(np.shape(img)))
    assert np.shape(padded_img) == (92, 92), "check zero pad function"
    return padded_img

def pad_images():
    new_dir = os.path.join(os.getcwd(), "new_integer_images")
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    imgs_dir = os.path.join(os.getcwd(), "integer_images")
    for fname in os.listdir(imgs_dir):
        f = os.path.join(imgs_dir, fname)
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if np.shape(img) != (92, 92):
            img = zero_pad_img(img)
        save_path = os.path.join(new_dir, fname)
        cv2.imwrite(save_path, img)


#########################################
#            Misc Functions             #
#########################################

def resize_images():
    imgs_dir = os.path.join(os.getcwd(), "integer_images_dataset")
    outdir = os.path.join(os.getcwd(), "integer_images_dataset_small")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for fname in os.listdir(imgs_dir):
        f = os.path.join(imgs_dir, fname)
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (38, 38))
        outfile = os.path.join(outdir, fname)
        cv2.imwrite(outfile, img)

def crop_img2d(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img1 = img[:32, 30:46]
    img2 = img[:32, 46:62]
    # img1 = img[:32, 26:38]
    # img2 = img[:32, 40:58]
    outdir = os.path.join(os.getcwd(), "newer_images")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outfile1 = os.path.join(outdir, "image7_10.png")
    outfile2 = os.path.join(outdir, "image9_6.png")
    cv2.imwrite(outfile1, img1)
    cv2.imwrite(outfile2, img2)

def crop_img3d(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img1 = img[:32, 24:34]
    img2 = img[:32, 34:52]
    img3 = img[:32, 52:68]
    outdir = os.path.join(os.getcwd(), "newer_images")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outfile1 = os.path.join(outdir, "image1_8.png")
    outfile2 = os.path.join(outdir, "image2_6.png")
    outfile3 = os.path.join(outdir, "image1_9.png")
    cv2.imwrite(outfile1, img1)
    cv2.imwrite(outfile2, img2)
    cv2.imwrite(outfile3, img3)
