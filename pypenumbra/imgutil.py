import cv2
import numpy as np
from skimage.draw import line
import math

def threshold(gray_image):
    """Uses a gaussian blur and Otsu method thresholding
    to achieve binary image.

    :param gray_image: An OpenCV grayscale image
    :returns: A binary image
    """

    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    retval, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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
    min_blob_area = (height*width)*0.05

    # Getting contour with the largest area larger than the min blob area
    contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    penumbra_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area and area > min_blob_area:
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

def get_line(x1, y1, x2, y2, image, radius):
    dx = x2-x1
    dy = y2-y1
    n = int(round(math.sqrt(dx*dx + dy*dy)))

    data = []
    xinc = float(dx/n)
    yinc = float(dy/n)
    rx = float(x1)
    ry = float(y1)
    for i in range(0, n):
      data.append(get_interpolated_value(rx, ry, image));
      rx += xinc;
      ry += yinc;
    
    return data

def get_interpolated_value(x, y, image):
    width = image.shape[1]
    height = image.shape[0]
    if x < 0.0 or x >= width - 1.0 or y < 0.0 or y >= height - 1.0:
        if x<-1.0 or x>=width or y<-1.0 or y>=height:
            return 0.0;
        else:
            print("Need edge pixels. Functionality not implemented.")
            return 0.0
            #return getInterpolatedEdgeValue(x, y, image);
    
    xbase = int(x);
    ybase = int(y);
    xFraction = x - xbase;
    yFraction = y - ybase;
    if xFraction < 0.0:
        xFraction = 0.0;
    if yFraction < 0.0:
        yFraction = 0.0;
    
    upperLeft = image[xbase, ybase]
    lowerLeft = image[xbase+1, ybase]
    upperRight = image[xbase+1, ybase+1]
    lowerRight = image[xbase, ybase+1]

    upperAverage = upperLeft + xFraction * (upperRight - upperLeft)
    lowerAverage = lowerLeft + xFraction * (lowerRight - lowerLeft)

    return lowerAverage + yFraction * (upperAverage - lowerAverage)

def apply_first_derivative(float_image):
    derivative_image = []
    for column in float_image.T:
        length = len(column)
        output_col = np.zeros_like(column)
        for i in range(1, (len(column) - 1)):
            output_col[i] = column[i-1] - column[i];
        derivative_image.append(output_col)
    return np.array(derivative_image)
