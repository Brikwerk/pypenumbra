import os

from . import sinogram
from skimage import io
from skimage import img_as_ubyte, img_as_float64
from skimage.transform import iradon
import numpy as np


def reconstruct_from_image(image_path):
    # Attempting to load an image in grayscale
    image = io.imread(image_path, as_gray=True)
    # Ensuring float and uint8 images are available
    float_image = img_as_float64(image)
    uint8_image = img_as_ubyte(image)

    return reconstruct(float_image, uint8_image)

def reconstruct_from_ctdata(data_path):
    # TODO: Load raw ct image data
    print("test")

def reconstruct(float_image, uint8_image):
    # Getting sinogram
    sinogram_image = sinogram.construct_sinogram(float_image, uint8_image)

    # Reconstructing the focal spot with filtered backprojection
    theta = np.linspace(0., 360., max(sinogram_image.shape), endpoint=False)
    focal_spot_image = iradon(sinogram_image, theta=theta, circle=True)

    return focal_spot_image, sinogram_image
