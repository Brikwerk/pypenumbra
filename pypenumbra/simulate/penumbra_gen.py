"""
    pypenumbra.simulate.penumbra_gen
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Defines the functionality for generating simulated
    penumbra images.
    :copyright: 2020 Reece Walsh
    :license: MIT
"""
import numpy as np
from skimage import io, img_as_ubyte
from skimage.exposure import equalize_adapthist
import cv2
from . import kernel_gen as kg


def generate_blank_penumbra_square(size, circle_radius):
    """Generates a numpy-based, OpenCV-drawn white circle of set radius
    on a black background of set size.
    
    :param size: How large the generated image is
    :type size: int
    :param circle_radius: How large the radius of the drawn circle is
    :type circle_radius: int
    :return: A numpy image with the drawn circle and black background
    :rtype: numpy.ndarray
    """

    img = np.zeros((size, size))
    cv2.circle(img, (size//2, size//2), circle_radius, (1.0,1.0,1.0), thickness=-1)

    return img


def generate_blank_penumbra_rectangle(width, height, circle_radius):
    """Generates a numpy-based, OpenCV-drawn white circle of set radius
    on a black background of a set width and height.
    
    :param width: The width of the generated image
    :type width: int
    :param height: The height of the generated image
    :type height: int
    :param circle_radius: The radius of the drawn circle
    :type circle_radius: int
    :return: A numpy image with the drawn circle and black background
    :rtype: numpy.ndarray
    """

    img = np.zeros((height, width))
    cv2.circle(img, (width//2, height//2), circle_radius, (1.0,1.0,1.0), thickness=-1)

    return img


def generate_blank_penumbra_cr18x24(circle_radius):
    """Generates a numpy-based, OpenCV drawn white circle of set radius
    on a black background of the size present on a CR 18"x24" cassette.
    
    :param circle_radius: The radius of a the drawn circle
    :type circle_radius: int
    :return: A numpy image with the drawn circle and black background
    :rtype: numpy.ndarray
    """

    return generate_blank_penumbra_rectangle(2370, 1770, circle_radius)


def generate_penumbra(blank_penumbra, kernel):
    """Convolves a supplied, blank penumbra image with a supplied
    filter kernel and returns the resulting penumbra image.
    
    :param blank_penumbra: A blank penumbra image (white circle on a
    black background)
    :type blank_penumbra: numpy.ndarray
    :param kernel: A filter kernel that the blank penumbra is convolved by
    :type kernel: numpy.ndarray
    :return: A numpy image containing the penumbra
    :rtype: numpy.ndarray
    """

    filter_img = cv2.filter2D(blank_penumbra, -1, kernel)
    img_stretched = img_as_ubyte(equalize_adapthist(filter_img))

    return img_stretched


if __name__ == "__main__":
    blank_penumbra = generate_blank_penumbra_cr18x24(246)
    kernel = kg.create_dual_point_kernel(69, 35)
    penumbra_img = generate_penumbra(blank_penumbra, kernel)
    io.imsave("../../penumbra-generated.png", penumbra_img)
