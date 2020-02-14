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
from skimage import exposure
from skimage.exposure import equalize_adapthist
from skimage import img_as_ubyte, img_as_float64
from skimage.transform import iradon
import numpy as np
import cv2

from matplotlib import pyplot as plt


def map_cr_values(binary_image, kvp=70):
    """Maps values from a binary CR image to a float image
    based off of a calculation involving the kVp used to
    generate the image.
    
    :param binary_image: A Numpy array containing the CR image values
    :type binary_image: numpy.ndarray
    :param kvp: The kVp used to generate the CR binary image, defaults to 70
    :type kvp: int, optional
    :return: A float image with values ranging from (-1, 1)
    :rtype: numpy.ndarray
    """

    # Getting C value for mapping equation
    # NOTE: This equation won't be an exact fit for most CR detectors,
    # however, it should be good enough for the purposes of
    # calibration within this library.
    C = (-0.0739 * np.power(kvp, 2)) + (15.408 * kvp) + 301.17
    # Applying CR mapping equation
    map_values = np.subtract(binary_image, C)
    map_values = np.divide(map_values, 1024)
    map_values = np.power(10, map_values)
    # Normalizing values to float image range (-1, 1)
    map_values = np.divide(map_values, np.max(map_values))

    return map_values


def reconstruct_from_image(image_path, debug=False):
    """Reconstructs the focal spot and the sinogram
    from a penumbra image specified by an image path.
    
    :param image_path: The path to the penumbra image
    :type image_path: string
    :param debug: A boolean value representing if debug images are output, defaults to False
    :type debug: bool, optional
    :return: A tuple containing the reconstructed image and the sinogram image.
    :rtype: (numpy.ndarray, numpy.ndarray)
    """

    # Attempting to load an image in grayscale
    image = io.imread(image_path, as_gray=True)
    # Ensuring float and uint8 images are available
    float_image = img_as_float64(image)
    uint8_image = img_as_ubyte(image)

    return reconstruct(float_image, uint8_image, debug=debug)


def reconstruct_from_array(image_array, debug=False):
    """Reconstructs the focal spot and the sinogram
    from a passed penumbra image in the form of an array.
    
    :param image_array: The penumbra image as a numpy array
    :type image_path: numpy.ndarray
    :param debug: A boolean value representing if debug images are output, defaults to False
    :type debug: bool, optional
    :return: A tuple containing the reconstructed image and the sinogram image.
    :rtype: (numpy.ndarray, numpy.ndarray)
    """

    # Ensuring float and uint8 images are available
    float_image = img_as_float64(image_array)
    uint8_image = img_as_ubyte(image_array)

    return reconstruct(float_image, uint8_image, debug=debug)


def reconstruct_from_cr_data(data_path, width, height, dtype="uint16", kvp=70, debug=False):
    """Reconstructs the focal spot and the sinogram
    from raw binary image specified by the data path.

    :param data_path: A path to the raw binary data
    :param width: The width of the binary image
    :param height: The height of the binary image
    :param dtype: The data type of the binary image
    :param kvp: The kVp used in the acquisition of the CR data
    :param debug: A boolean value representing if debug images are output
    :returns: A tuple containing the focal spot image
    and the sinogram image.
    """

    image = np.fromfile(data_path, dtype=dtype)
    image = image.reshape(width, height)
    image = map_cr_values(image, kvp=kvp)
    image = equalize_adapthist(image)
    # Ensuring float and uint8 images are available
    float_image = img_as_float64(image)
    uint8_image = img_as_ubyte(image)

    return reconstruct(float_image, uint8_image, debug=debug)


def reconstruct(float_image, ubyte_image, debug=False):
    """Reconstructs the focal spot and the sinogram
    from a penumbra image in float64 and uint8 format.

    :param float_image: The penumbra image in float64 format
    :param uint8_image: The penumbra image in uint8 format
    :param debug: A boolean value representing if debug images are output
    :returns: A tuple containing the focal spot image
    and the sinogram image.
    """

    # Getting sinogram
    sinogram_image = sinogram.construct_sinogram(float_image, ubyte_image, debug=debug)

    # Reconstructing the focal spot with filtered backprojection
    theta = np.linspace(0., 360., sinogram_image.shape[1], endpoint=False)
    focal_spot_image = iradon(sinogram_image, theta=theta, filter="ramp", circle=True)

    return focal_spot_image, sinogram_image
