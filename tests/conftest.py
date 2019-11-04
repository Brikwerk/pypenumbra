import pytest
from skimage import io
import cv2

@pytest.fixture
def penumbra_circle():
    image = io.imread("./tests/data/penumbra_test_circle.tif", as_gray=True)
    return image

@pytest.fixture
def focal_spot_circle():
    image = io.imread("./tests/data/focal_spot_circle.png", as_gray=True)
    return image

@pytest.fixture
def sinogram_circle():
    image = io.imread("./tests/data/sinogram_circle.png", as_gray=True)
    return image
