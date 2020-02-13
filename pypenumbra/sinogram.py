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
from skimage.exposure import equalize_adapthist
from skimage import img_as_ubyte
from skimage.transform import rotate
import math
import cv2

from . import imgutil


def construct_sinogram(float_image, uint8_image, angular_steps=360, debug=False):
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

    if debug:
        disk_lines = cv2.cvtColor(uint8_image, cv2.COLOR_GRAY2RGB)
        cv2.line(disk_lines, (center_x, center_y), (center_x+radius, center_y), (0, 255, 0), thickness=3)
        cv2.circle(disk_lines, (center_x, center_y), 5, (0, 255, 0), thickness=5)

        imgutil.save_debug_image("1 - original_image.png", uint8_image)
        imgutil.save_debug_image("2 - threshold_raw_image.png", threshold)
        imgutil.save_debug_image("3 - disk_stats.png", disk_lines)
        print("---")
        print("Original Image Disk Identification Results:")
        print("Center X: %d | Center Y: %d | Radius: %d" % (center_x, center_y, radius))

    if radius < 1:
        raise ValueError("Radius is of improper length")

    PADDING = int(round(radius * 0.323232))  # Relative padding
    # Dictates how large the area around the penumbra is when cropping
    # Also dictates the ultimate x/y size of the focal spot output
    radius = radius + PADDING # Padding radius

    # Slicing penumbra blob into sinogram
    sinogram = slice_penumbra_blob(center_x, center_y, radius, angular_steps, float_image, uint8_image, debug=debug)
    top, bottom, center = get_sinogram_size(sinogram, PADDING, debug=debug)

    if debug:
        rs_height, rs_width = sinogram.shape
        rs_lines = cv2.cvtColor(img_as_ubyte(sinogram), cv2.COLOR_GRAY2RGB)
        cv2.line(rs_lines, (0, top), (rs_width, top), (0, 255, 0), thickness=3)
        cv2.line(rs_lines, (0, center), (rs_width, center), (0, 255, 0), thickness=3)
        cv2.line(rs_lines, (0, bottom), (rs_width, bottom), (0, 255, 0), thickness=3)

        imgutil.save_debug_image("5 - radial_slices.png", img_as_ubyte(sinogram))
        imgutil.save_debug_image("7 - sinogram_lines.png", rs_lines)
        print("---")
        print("Sinogram Identification Stats:")
        print("Top of Sinogram: %d | Center of Sinogram: %d | Bottom of Sinogram: %d" % (top, center, bottom))

    # Applying first derivative on the vertical direction
    derivative_sinogram = imgutil.apply_first_derivative(sinogram)
    if debug:
        imgutil.save_debug_image("8 - derivative_sinogram.png", derivative_sinogram)

    # Cropping image around sinogram's center axis
    height, width = derivative_sinogram.shape
    crop_sinogram = derivative_sinogram[top:bottom, 0:width]

    return crop_sinogram


def slice_penumbra_blob(center_x, center_y, radius, angular_steps, float_image, uint8_image, debug=False):
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

    if debug:
        drawn_sino = img_as_ubyte(equalize_adapthist(float_image))
        drawn_sino = cv2.cvtColor(drawn_sino, cv2.COLOR_GRAY2RGB)

    # Assembling sinogram slices from the image
    for i in range(0, angular_steps):
        # Rotating around the penumbra blob in a circle by RADS_PER_SLICE
        angle = i * RADS_PER_SLICE
        outer_x = center_x + radius * math.cos(angle)
        outer_y = center_y - radius * math.sin(angle)

        if debug:
            drawn_sino = cv2.line(drawn_sino,(center_x, center_y),(int(round(outer_x)), int(round(outer_y))),(0,255,0),1)

        col = imgutil.get_line(center_x, center_y, outer_x, outer_y, float_image)
        sinogram[i] = col

    if debug:
        imgutil.save_debug_image("4 - slice_lines.png", drawn_sino)
    
    sinogram = np.rot90(sinogram, axes=(1,0))
    return sinogram


def get_sinogram_size(sinogram_input, padding, debug=False):
    """Gets the top, bottom, and center Y-coordinate

    :param sinogram_input: A sinogram image
    :param padding: How much padding (in pixels) to put around top/bottom Y-coordinates
    :returns: The top, bottom, and center of the sinogram as integers
    """

    sinogram = img_as_ubyte(sinogram_input)
    # Attempting to isolate the circle within the pre-sinogram image
    bsize = 15
    blur = cv2.bilateralFilter(sinogram, bsize, bsize*2, bsize/2)
    ret, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

    if debug:
        imgutil.save_debug_image("6 - threshold_sinogram.png", thresh)

    # Iterating over columns to find the center/radius
    # of the pre-sinogram. We assume that the black portion
    # of the pre-sinogram is on the bottom.
    thresh_rotate = rotate(thresh, -90, resize=True)
    height, width = thresh_rotate.shape
    sinogram_top = math.inf
    sinogram_center = 0
    # Iterating over each col (or row now, since we rotated)
    for col_idx in range(0, height):
        col = thresh_rotate[col_idx]
        sinogram_col_length = 0
        # Iterate through each value and break upon finding
        # the sinogram's edge
        for value_idx in range(0, len(col)):
            value = col[value_idx]
            if value > 0:
                sinogram_col_length = width - value_idx
                break
        # Update if the edge is a new low or high
        if sinogram_col_length != 0:
            if sinogram_col_length < sinogram_top:
                sinogram_top = sinogram_col_length
            elif sinogram_col_length > sinogram_center:
                sinogram_center = sinogram_col_length

    sinogram_radius = sinogram_center - sinogram_top
    sinogram_bottom = sinogram_center + sinogram_radius

    top = int(round(sinogram_top - padding))
    if top < 0:
        top = 0
    bottom = int(round(sinogram_bottom + padding))

    return top, bottom, sinogram_center
