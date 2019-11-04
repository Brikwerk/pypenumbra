import pypenumbra.api as api
from skimage import img_as_float64, img_as_ubyte
import utils

def test_reconstruct(penumbra_circle, focal_spot_circle, sinogram_circle):
    float_image = img_as_float64(penumbra_circle)
    uint8_image = img_as_ubyte(penumbra_circle)
    focal_spot, sinogram = api.reconstruct(float_image, uint8_image)

    fs_check = utils.duplicate_grayimage_check(img_as_ubyte(focal_spot), focal_spot_circle)
    sino_check = utils.duplicate_grayimage_check(img_as_ubyte(sinogram), sinogram_circle)
    
    assert fs_check and sino_check

def test_reconstruction_from_image(penumbra_circle, focal_spot_circle, sinogram_circle):
    focal_spot, sinogram = api.reconstruct_from_image("./tests/data/penumbra_test_circle.tif")

    fs_check = utils.duplicate_grayimage_check(img_as_ubyte(focal_spot), focal_spot_circle)
    sino_check = utils.duplicate_grayimage_check(img_as_ubyte(sinogram), sinogram_circle)
    
    assert fs_check and sino_check