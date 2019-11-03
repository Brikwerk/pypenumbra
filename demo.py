import pypenumbra
from skimage import io, img_as_ubyte
from skimage.exposure import equalize_adapthist
import os

focal_spot, sinogram = pypenumbra.reconstruct_from_image("./pypenumbra/test/data/penumbra_test_square.tif")
if not os.path.isdir("./results"):
    os.makedirs("./results")
io.imsave("./results/focal_spot.png", img_as_ubyte(equalize_adapthist(focal_spot)))
io.imsave("./results/sinogram.png", img_as_ubyte(equalize_adapthist(sinogram)))
