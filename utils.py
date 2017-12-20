import scipy.misc
import numpy as np


def get_image(image_path, image_size, is_crop=True):
    """

    :param image_path: string, image path
    :param image_size: int, image size that you want to crop
    :param is_crop: boolean, whether image cropped or not
    :return: float ndarray, each element within [-1, 1]
    if is_crop: shape is [npx, npx, 3]
    else: shape is [image_height, image_width, 3]
    """
    return transform(imread(image_path), image_size, is_crop)


def imread(path):
    """

    :param path: string, image path
    :return: float ndarray with dimension of [image_height, image_width, 3]
    """
    return scipy.misc.imread(path, mode='RGB').astype(np.float)


def transform(image, npx=64, is_crop=True):
    """
    This function used to transform each pixel from [0, 255] to [-1, 1]
    :param image: float ndarray with dimension of [image_height, image_width, 3],
    the image matrix, each pixel range within [0, 255]
    :param npx: int, image width/height size
    :param is_crop: boolean, image cropped or not
    :return: float ndarray, each element within [-1, 1]
    if is_crop: shape is [npx, npx, 3]
    else: shape is [image_height, image_width, 3]
    """
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.


def center_crop(x, crop_h, crop_w=None, resize_w=64):
    """

    :param x: float ndarray with dimension of [image_height, image_width, 3],
    the image matrix, each pixel range within [0, 255]
    :param crop_h: int, cropped image height size
    :param crop_w: int, cropped image width size
    :param resize_w: int, cropped image width/height size
    :return: float ndarray with dimension of [resize_w, resize_w, 3], a center cropped image
    """
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_w, resize_w])


def save_images(images, size, image_path):
    """

    :param images: 4D-tensor, shape is [batch, height, width, channels]
    :param size: int list, length is 2, denote the number of sample images
    :param image_path: string, save image path
    :return:
    """
    return imsave(inverse_transform(images), size, image_path)


def inverse_transform(images):
    return (images+1.)/2.


def imsave(images, size, path):
    img = merge(images, size)
    return scipy.misc.imsave(path, (255*img).astype(np.uint8))


def merge(images, size):
    """

    :param images: 4D-tensor, shape is [batch, height, width, channels]
    :param size: int list, length is 2, sample images number
    :return: float ndarray, shape is [height*size[0], width*size[1], 3], merged images
    """
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

