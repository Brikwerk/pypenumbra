"""
    pypenumbra.sinogram
    ~~~~~~~~~~~~~~
    Defines the logic to enable the construction of a sinogram
    from a penumbra blob.
    :copyright: 2019 Reece Walsh
    :license: MIT
"""
import math

import numpy as np
import cv2

from . import imgutil


def construct_sinogram(float_image, uint8_image, angular_steps=360):
    """Constructs a sinogram from the detected penumbra blob
    in the passed images. The uint8 image is used for blob detection
    and the float image is used for value calculations.

    :param float_image: A float64 image used for value calculations
    :param uint8_image: A uint8 image used for blob detection
    :param angular_steps: The number of slices to slice the blob into
    :returns: A float64 sinogram image
    """

    # Detecting penumbra blob and getting properties
    threshold = imgutil.threshold(uint8_image)
    center_x, center_y, radius = imgutil.get_center(threshold)
    PADDING = int(round(radius * 0.323232))  # Relative padding
    # Dictates how large the area around the penumbra is when cropping
    # Also dictates the ultimate x/y size of the focal spot output
    radius = radius + PADDING # Padding radius

    # Slicing penumbra blob into sinogram
    sinogram = slice_penumbra_blob(center_x, center_y, radius, angular_steps, float_image)

    # Applying first derivative on the vertical direction
    derivative_sinogram = imgutil.apply_first_derivative(sinogram)

    # Cropping image to 64 pixels around sinogram
    upper_slice = int(round(PADDING*1.5))
    lower_slice = int(round(PADDING/2))
    crop_sinogram = derivative_sinogram[radius-upper_slice:radius-lower_slice, 0:angular_steps]

    return crop_sinogram


def slice_penumbra_blob(center_x, center_y, radius, angular_steps, float_image):
    """Slices a penumbra blob into a specified number of slices

    :param center_x: The x-coordinate of the center of the penumbra blob
    :param center_y: The y-coordinate of the center of the penumbra blob
    :param radius: The radius of the penumbra blob (can include padding)
    :param angular_steps: How many slices to slice the blob into
    :param float_image: A float64 image used to source the slices from
    :returns: Slices compiled into an image
    """

    # Setting up values for sinogram extraction
    ARC_ANGLE = 360.0
    RADS_PER_SLICE = (math.pi/180.0) * (ARC_ANGLE/angular_steps)
    sinogram = np.zeros(shape=(angular_steps, radius), dtype="float64")
    outer_x = 0
    outer_y = 0

    # Assembling sinogram slices from the image
    for i in range(0, angular_steps):
        # Rotating around the penumbra blob in a circle by RADS_PER_SLICE
        angle = i * RADS_PER_SLICE
        outer_x = center_x + radius * math.cos(angle)
        outer_y = center_y - radius * math.sin(angle)
        col = imgutil.get_line(center_x, center_y, outer_x, outer_y, float_image)
        sinogram[i] = col
    
    sinogram = np.rot90(sinogram, axes=(1,0))
    return sinogram
