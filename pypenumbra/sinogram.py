import math

import numpy as np
import cv2

from . import imgutil


def construct_sinogram(float_image, uint8_image, angular_steps=360):
    # Detecting penumbra blob and getting properties
    threshold = imgutil.threshold(uint8_image)
    center_x, center_y, radius = imgutil.get_center(threshold)

    # Setting up values for sinogram extraction
    ARC_ANGLE = 360.0
    RADS_PER_SLICE = (math.pi/180.0) * (ARC_ANGLE/angular_steps)
    # Dictates how large the area around the penumbra is when cropping
    # Also dictates the ultimate x/y size of the focal spot output
    # PADDING = int(round(radius * 0.323232))  # Relative padding
    PADDING = 64
    radius = radius + PADDING # Padding radius

    sinogram = np.zeros(shape=(angular_steps, radius), dtype="float64")
    outer_x = 0
    outer_y = 0

    # Assembling sinogram slices from the image
    for i in range(0, angular_steps):
        # Rotating around the penumbra blob in a circle by RADS_PER_SLICE
        angle = i * RADS_PER_SLICE
        outer_x = center_x + radius * math.cos(angle)
        outer_y = center_y - radius * math.sin(angle)
        col = imgutil.get_line(center_x, center_y, outer_x, outer_y, float_image, radius)
        sinogram[i] = col
    
    sinogram = np.rot90(sinogram, axes=(1,0))
    # Applying first derivative on the vertical direction
    derivative_sinogram = imgutil.apply_first_derivative(sinogram)
    derivative_sinogram = np.rot90(derivative_sinogram, axes=(1,0))
    # Cropping image to 64 pixels around sinogram
    upper_slice = int(round(PADDING*1.5))
    lower_slice = int(round(PADDING/2))
    crop_sinogram = derivative_sinogram[radius-upper_slice:radius-lower_slice, 0:angular_steps]

    return crop_sinogram
