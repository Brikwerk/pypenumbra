"""
    pypenumbra.imgutil
    ~~~~~~~~~~~~~~
    Defines the utility functions for working with images.
    :copyright: 2019 Reece Walsh
    :license: MIT
"""
import os
import math

import cv2
import numpy as np
from skimage import filters, img_as_ubyte
from skimage.exposure import equalize_adapthist
from skimage.io import imsave


def save_debug_image(image_name, image):
    """Saves an image into the debug_images directory

    :param image_name: The name to store the image as, extension included
    :param image: An image
    """

    if not os.path.isdir("./debug_images"):
        os.mkdir("./debug_images")
    image_path = os.path.join("./debug_images", image_name)
    imsave(image_path, img_as_ubyte(equalize_adapthist(image)))


def threshold(gray_image):
    """Uses a gaussian blur and Otsu method thresholding
    to achieve binary image.

    :param gray_image: An OpenCV grayscale image
    :returns: A binary image
    """

    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    retval, threshold = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY)

    return threshold


def get_center(threshold_image):
    """Selects the largest detected blob within an image,
    returns the center x and y coordinates as a tuple, and the radius
    of the blob.

    :param threshold_image: An OpenCV image with thresholding applied
    :returns: The coordinates of the center and radius of the largest blob
    """

    # Get rough min area of blob
    # Penumbra must be at least 5% of the
    # total image size to be detected
    height = threshold_image.shape[0]
    width = threshold_image.shape[1]

    # Getting contour with the largest area larger than the min blob area
    contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    penumbra_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            penumbra_contour = contour
            max_area = area

    if penumbra_contour is None:
        center_x = 0
        center_y = 0
        radius = 0
    else:
        # Getting the center x,y of the penumbra blob
        M = cv2.moments(penumbra_contour)
        center_x = int(M['m10']/M['m00'])
        center_y = int(M['m01']/M['m00'])

        # Getting radius of the penumbra blob
        (circle_x, circle_y), radius = cv2.minEnclosingCircle(penumbra_contour)
        radius = int(radius)

    return (center_x, center_y, radius)


def pad_to_fit(radius, center_x, center_y, image):
    height, width = image.shape
    pad_amount = 0

    top = center_y - radius
    if top < 0:
        pad_amount = abs(top)

    right = (center_x + radius) - width
    if right > 0 and right > pad_amount:
        pad_amount = right
    
    bottom = (center_y + radius) - height
    if bottom > 0 and bottom > pad_amount:
        pad_amount = bottom
    
    left = center_x - radius
    if left < 0 and abs(left) > pad_amount:
        pad_amount = left
    
    pad_image = np.pad(image, pad_amount+1)

    return center_x + pad_amount, center_y + pad_amount, pad_image


def get_line(x1, y1, x2, y2, image):
    """Gets a line of pixels from the center of an image outwards
    to a specified point. If a subpixel is specified during traversal to
    the outward point, bilinear interpolation is used to get the value.

    :param x1: The center x-coordinate
    :param y1: The center y-coordinate
    :param x2: The outward x-coordinate
    :param y2: The outward y-coordinate
    :param image: The float64 image used to get pixel values from
    :returns: An array of pixel values between the two points
    """
    
    dx = x2-x1
    dy = y2-y1
    n = int(round(math.sqrt(dx*dx + dy*dy)))

    data = []
    x_increment = float(dx/n)
    y_increment = float(dy/n)
    rx = float(y1)
    ry = float(x1)
    for i in range(0, n):
        value = bilinear_interpolate(rx, ry, image)
        data.append(value)
        rx += x_increment
        ry += y_increment
    
    return data


def bilinear_interpolate(x, y, image):
    """Utilize bilinear resampling to get the x/y pixel value in the image

    :param x: An x coordinate within the image in float format
    :param x: A y coordinate within the image in float format
    :param image: An image in float64 format
    """

    x1 = int(x)
    x2 = x1 + 1
    y1 = int(y)
    y2 = y1 + 1

    upper_left = image[x1, y1]
    lower_left = image[x1, y2]
    upper_right = image[x2, y1]
    lower_right = image[x2, y2]

    upper_average = ((x2 - x)/(x2 - x1) * upper_left) + ((x - x1)/(x2 - x1) * upper_right)
    lower_average = ((x2 - x)/(x2 - x1) * lower_left) + ((x - x1)/(x2 - x1) * lower_right)

    return ((y2 - y)/(y2 - y1) * upper_average) + ((y - y1)/(y2 - y1) * lower_average)


def apply_first_derivative(float_image):
    """Computes the first derivative of the image with the Prewitt filter.

    :param float_image: A float64 format image
    :returns: A float64 format image
    """

    edges = filters.scharr(float_image)
    #edges = filters.prewitt(float_image)

    # derivative_image = []
    # for column in float_image.T:
    #     length = len(column)
    #     output_col = np.zeros_like(column)
    #     for i in range(1, (len(column) - 1)):
    #         output_col[i] = column[i-1] - column[i]
    #     derivative_image.append(output_col)
    # derivative_image = np.array(derivative_image)
    # derivative_image = np.rot90(derivative_image, axes=(1,0))

    return edges
