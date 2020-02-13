import pypenumbra
import numpy as np
from skimage import io, img_as_ubyte
from skimage.exposure import equalize_adapthist
import os

# If we want debug images in the debug_images folder or not
# NOTE: Disabling speeds up the reconstruction phase massively
DEBUG = True

# IMAGE_PATH = "./example_images/high_overlap_dual_point_source_simulated.png"
# IS_BINARY = False
IMAGE_PATH = "./example_images/high_overlap_dual_point_source_binary.std"
IS_BINARY = True

if IS_BINARY:
    # Saving raw image as png
    image = np.fromfile(IMAGE_PATH, dtype="uint16")
    image = image.reshape(1770, 2370)
    map_image = pypenumbra.map_cr_values(image, kvp=70)
    # Ensuring float and uint8 images are available
    io.imsave("./results/raw_image.png", img_as_ubyte(equalize_adapthist(map_image)))

if IS_BINARY:
    focal_spot, sinogram = pypenumbra.reconstruct_from_cr_data(IMAGE_PATH, 1770, 2370, debug=DEBUG)
else:
    focal_spot, sinogram = pypenumbra.reconstruct_from_image(IMAGE_PATH, debug=DEBUG)

if not os.path.isdir("./results"):
    os.makedirs("./results")
io.imsave("./results/reconstruction_image.png", img_as_ubyte(equalize_adapthist(focal_spot)))
io.imsave("./results/sinogram.png", img_as_ubyte(equalize_adapthist(sinogram)))
