import numpy as np

from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave, imshow
from skimage.transform import radon, rescale
from skimage.exposure import equalize_adapthist

from skimage.transform import iradon

from skimage.transform import iradon_sart, rotate

theta = np.linspace(0., 360., 360, endpoint=False)
sinogram = img_as_float(imread("./results/sinogram.png"))

reconstruction_fbp = iradon(sinogram, theta=theta, circle=True)

art_fbp_sinogram = radon(reconstruction_fbp, theta=theta, circle=True)
imsave("./results/reconstruction_fbp.png", img_as_ubyte(equalize_adapthist(reconstruction_fbp)))

reconstruction_sart = iradon_sart(sinogram, theta=theta)

reconstruction_sart2 = iradon_sart(sinogram, theta=theta,
                                   image=reconstruction_sart)

ITERATIONS = 5
for i in range(0, ITERATIONS):
    reconstruction_sart2 = iradon_sart(sinogram, theta=theta,
                                   image=reconstruction_sart2)

imsave("./results/reconstruction_art.png", img_as_ubyte(equalize_adapthist(reconstruction_sart2)))
