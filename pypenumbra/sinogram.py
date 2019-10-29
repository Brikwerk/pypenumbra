from . import imgutil
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

def construct_sinogram(float_image, uint8_image, angular_steps=360):
    # Detecting penumbra blob and getting properties
    threshold = imgutil.threshold(uint8_image)
    center_x, center_y, radius = imgutil.get_center(threshold)

    # Setting up values for sinogram extraction
    ARC_ANGLE = 360.0
    RADS_PER_SLICE = (math.pi/180.0) * (ARC_ANGLE/angular_steps)
    radius = radius + 64 # Padding radius
    sinogram = np.zeros(shape=(angular_steps, radius), dtype="float64")
    outer_x = 0
    outer_y = 0

    # Assembling sinogram slices from the image
    for i in range(0, angular_steps):
        angle = i * RADS_PER_SLICE
        outer_x = center_x + radius * math.cos(angle)
        outer_y = center_y - radius * math.sin(angle)
        col = imgutil.get_line(center_x, center_y, outer_x, outer_y, float_image, radius)
        sinogram[i] = col
    
    sinogram = np.rot90(sinogram, axes=(1,0))
    # Applying first derivative on the vertical direction
    derivative_sinogram = imgutil.apply_first_derivative(sinogram)
    derivative_sinogram = np.rot90(derivative_sinogram, axes=(1,0))
    # Cropping image
    crop_sinogram = derivative_sinogram[radius-96:radius, 0:angular_steps]
    
    cv2.imshow("Im", crop_sinogram)
    cv2.waitKey(0)
