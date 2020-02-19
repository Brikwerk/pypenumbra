import os

from skimage import img_as_float64, img_as_ubyte, io
from skimage.exposure import equalize_adapthist

import pypenumbra as ppen
from pypenumbra import simulate as psim

# If we want debug images in the debug_images folder or not
# NOTE: Disabling speeds up the reconstruction phase massively
DEBUG = True

# Generating simulation data with CR template
blank_penumbra = psim.generate_blank_penumbra_cr18x24(246)
kernel = psim.create_dual_point_kernel(69, 35)
penumbra_img = psim.generate_penumbra(blank_penumbra, kernel)

# Reconstructing
focal_spot, sinogram = ppen.reconstruct_from_array(penumbra_img, debug=DEBUG)

# Saving results
if not os.path.isdir("./results"):
    os.makedirs("./results")
io.imsave("./results/reconstruction_image.png", img_as_ubyte(equalize_adapthist(focal_spot)))
io.imsave("./results/sinogram.png", img_as_ubyte(equalize_adapthist(sinogram)))