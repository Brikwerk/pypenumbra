"""
    pypenumbra.cli
    ~~~~~~~~~~~~~~
    Defines all logic for the pypenumbra command line interface
    application.
    :copyright: 2019 Reece Walsh
    :license: MIT
"""
import os

import fire
from skimage import io, img_as_ubyte
from skimage.exposure import equalize_adapthist

from .api import reconstruct_from_image, reconstruct_from_binary


class PyPenumbraCLI():
    """
    pypenumbra.cli

    A Python command line application for the reconstruction of
    focal spot and sinogram images from a penumbra image.

    Commands:
        reconstruct - Reconstructs a focal spot/sinogram image from
        a penumbra in a referenced image.

    Please type "pypenumbra COMMAND --help" for more information
    about these commands.
    """

    def image_reconstruct(self, data_path, output_dir="",
    focal_spot_image_name="focal_spot", sinogram_image_name="sinogram"):
        """Reconstructs a focal spot/sinogram image from
        a penumbra in a referenced image.

        :param data_path: The path to an image or raw data image of a penumbra
        :param output_dir: The path to a directory to save the output images
        :param focal_spot_image_name: The name of the focal spot image
        :param sinogram_image_name: The name of the sinogram image
        """

        focal_spot, sinogram = reconstruct_from_image(data_path)

        focal_spot = img_as_ubyte(equalize_adapthist(focal_spot))
        focal_spot_path = os.path.join(output_dir, "%s.png" % focal_spot_image_name)
        sinogram = img_as_ubyte(equalize_adapthist(sinogram))
        sinogram_path = os.path.join(output_dir, "%s.png" % sinogram_image_name)

        io.imsave(focal_spot_path, focal_spot)
        io.imsave(sinogram_path, sinogram)
    
    def binary_reconstruct(self, data_path, width, height, dtype="uint16",
    output_dir="", focal_spot_image_name="focal_spot", 
    sinogram_image_name="sinogram"):
        """Reconstructs a focal spot/sinogram image from
        a penumbra in a referenced binary image.

        :param data_path: The path to an image or raw data image of a penumbra
        :param width: The width of the binary image
        :param height: The height of the binary image
        :param dtype: The data type of the binary image
        :param output_dir: The path to a directory to save the output images
        :param focal_spot_image_name: The name of the focal spot image
        :param sinogram_image_name: The name of the sinogram image
        """

        focal_spot, sinogram = reconstruct_from_binary(data_path, width, height, dtype=dtype)

        focal_spot = img_as_ubyte(equalize_adapthist(focal_spot))
        focal_spot_path = os.path.join(output_dir, "%s.png" % focal_spot_image_name)
        sinogram = img_as_ubyte(equalize_adapthist(sinogram))
        sinogram_path = os.path.join(output_dir, "%s.png" % sinogram_image_name)

        io.imsave(focal_spot_path, focal_spot)
        io.imsave(sinogram_path, sinogram)


def main():
    fire.Fire(PyPenumbraCLI)


if __name__ == "__main__":
    main()
