"""
    pypenumbra.api
    ~~~~~~~~~~~~~~
    Defines the API for pypenumbra.
    :copyright: 2019 Reece Walsh
    :license: MIT
"""
import os

from . import sinogram
from skimage import io
from skimage import img_as_ubyte, img_as_float64
from skimage.transform import iradon
import numpy as np


def reconstruct_from_image(image_path):
    """Reconstructs the focal spot and the sinogram
    from a penumbra image specified by an image path.

    :param image_path: A path to the penumbra image
    :returns: A tuple containing the focal spot image
    and the sinogram image.
    """

    # Attempting to load an image in grayscale
    image = io.imread(image_path, as_gray=True)
    # Ensuring float and uint8 images are available
    float_image = img_as_float64(image)
    uint8_image = img_as_ubyte(image)

    return reconstruct(float_image, uint8_image)


def reconstruct_from_binary(data_path, width, height, dtype="uint16"):
    """Reconstructs the focal spot and the sinogram
    from raw binary image specified by the data path.

    :param data_path: A path to the raw binary data
    :param width: The width of the binary image
    :param height: The height of the binary image
    :param dtype: The data type of the binary image
    :returns: A tuple containing the focal spot image
    and the sinogram image.
    """

    image = np.fromfile(data_path, dtype=dtype)
    image = image.reshape(width, height)
    # Ensuring float and uint8 images are available
    float_image = img_as_float64(image)
    uint8_image = img_as_ubyte(image)

    return reconstruct(float_image, uint8_image)


def reconstruct(float_image, uint8_image):
    """Reconstructs the focal spot and the sinogram
    from a penumbra image in float64 and uint8 format.

    :param float_image: The penumbra image in float64 format
    :param uint8_image: The penumbra image in uint8 format
    :returns: A tuple containing the focal spot image
    and the sinogram image.
    """

    # Getting sinogram
    sinogram_image = sinogram.construct_sinogram(float_image, uint8_image)

    # Reconstructing the focal spot with filtered backprojection
    theta = np.linspace(0., 360., max(sinogram_image.shape), endpoint=False)
    focal_spot_image = iradon(sinogram_image, theta=theta, circle=True)

    return focal_spot_image, sinogram_image
