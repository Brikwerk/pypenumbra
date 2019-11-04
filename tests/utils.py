import cv2

def duplicate_grayimage_check(original, check):
    difference = cv2.subtract(original, check)
    if cv2.countNonZero(difference) == 0:
        return True
    else:
        return False
