"""
    pypenumbra.simulate.kernel_gen
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Defines the functionality for generating simulated
    filter kernels.
    :copyright: 2020 Reece Walsh
    :license: MIT
"""
import numpy as np
from skimage import io, img_as_float


def create_square_kernel(size, intensity, padding=0):
    """Generates a float image of a set intensity in the shape of a square.
    
    :param size: The size of the float image
    :type size: int
    :param intensity: The intensity of the square float image (0-255)
    :type intensity: int
    :param padding: How much to pad the float image (zero-padded), defaults to 0
    :type padding: int, optional
    :raises ValueError: If intensity or size is of an improper value
    :return: A 2D numpy array containing the square float image
    :rtype: numpy.ndarray
    """

    if intensity > 255 or intensity < 0:
        raise ValueError("Intensity must be <255 and >0")
    if size < 0:
        raise ValueError("Size must be of type int and >=0")

    kernel = np.empty((size, size))
    kernel.fill(intensity/255)
    kernel = np.pad(kernel, padding)

    return kernel


def create_rectangle_kernel(width, height, intensity, padding=0):
    """Generates a float image of a set width, height, and intensity.
    
    :param width: The width of the float image
    :type width: int
    :param height: The height of the float image
    :type height: int
    :param intensity: The intensity of the float image
    :type intensity: int
    :param padding: How much to pad the float image (zero-padded), defaults to 0
    :type padding: int, optional
    :raises ValueError: If intensity or size is of an improper value
    :return: A 2D numpy array containing the rectangular float image
    :rtype: numpy.ndarray
    """

    if intensity > 255 or intensity < 0:
        raise ValueError("Intensity must be <255 and >0")
    if size < 0:
        raise ValueError("Size must be of type int and >=0")

    kernel = np.empty((height, width))
    kernel.fill(intensity/255)
    kernel = np.pad(kernel, padding)

    return kernel


def create_kernel_from_image(image_path, padding=0):
    """Generates a float image from an image on the disk.
    
    :param image_path: The path to the reference image
    :type image_path: string
    :param padding: How much to pad the float image (zero-padded), defaults to 0
    :type padding: int, optional
    :return: A 2D numpy array containing the float image
    :rtype: numpy.ndarray
    """

    img = io.imread(image_path, as_gray=True)
    img = img_as_float(img)
    kernel = np.pad(img, padding)

    return kernel


def create_dual_point_kernel(size, distance_apart, padding=0):
    """Generates a square float image with two, single-pixel point sources.
    The top source has double the intensity of the bottom source. The sources
    are distributed along the vertical center and are spaced a set distance apart.
    
    :param size: The overall size of the float image, must be odd
    :type size: int
    :param distance_apart: How far apart (in pixels) the sources are
    :type distance_apart: int
    :param padding: How much to pad the float image (zero-padded), defaults to 0
    :type padding: int, optional
    :raises ValueError: If size and/or distance_apart are of an improper value
    :return: A 2D numpy array containing the float image
    :rtype: numpy.ndarray
    """

    if size < 0:
        raise ValueError("size must be of type int and >=0")
    if size % 2 == 0:
        raise ValueError("size must be odd")
    if distance_apart % 2 == 0:
        raise ValueError("distance_apart must be odd")
    if size <= distance_apart:
        raise ValueError("size must be >distance_apart")
    kernel = np.zeros((size, size))
    middle = size//2 # Floor dividing
    radius_apart = distance_apart//2 + 1
    kernel[middle, middle-radius_apart] = 0.25
    kernel[middle, middle+radius_apart] = 0.5
    kernel = np.pad(kernel, padding)

    return kernel


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    io.imshow(create_dual_point_kernel(69, 47))
    plt.show()