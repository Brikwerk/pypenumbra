# PyPenumbra

This project aims to re-produce and re-implement the methods published in the paper 
"X-ray Focal Spot Reconstruction by Circular Penumbra Analysis" by Dr. Giovanni Di Domenico et al.

## Getting Started

To begin, either clone the project and install

```bash

    pip install .

```

Or install directly with pip

```bash

    pip install git+https://github.com/brikwerk/pypenumbra

```

PyPenumbra comes with an API and a CLI

## API

The API extends two main functions: "reconstruct_from_image" and "reconstruct_from_cr_data".

The image reconstruction function requires a path to an image and outputs the focal spot image
and the sinogram in float64 format.

```python

    import pypenumbra
    from skimage import io, img_as_ubyte
    from skimage.exposure import equalize_adapthist
    import os

    focal_spot, sinogram = pypenumbra.reconstruct_from_image("image.tif")
    io.imsave("focal_spot.png", img_as_ubyte(equalize_adapthist(focal_spot)))
    io.imsave("sinogram.png", img_as_ubyte(equalize_adapthist(sinogram)))

```

The binary reconstruction function requires the path to the data, the width of the data,
the height of the data, and the data type. The focal spot image and the sinogram image
are output in float 64 format. This function would be used to reconstruct from raw CR 
or DR data.

```python

    import pypenumbra
    from skimage import io, img_as_ubyte
    from skimage.exposure import equalize_adapthist
    import os

    focal_spot, sinogram = pypenumbra.reconstruct_from_cr_data("image.std", 2370, 1770, dtype="uint16")
    io.imsave("focal_spot.png", img_as_ubyte(equalize_adapthist(focal_spot)))
    io.imsave("sinogram.png", img_as_ubyte(equalize_adapthist(sinogram)))

```

## CLI

Once PyPenumbra has been installed with pip, reconstruction from images and binary images is made
available on the command line.

```bash

    pypenumbra image_reconstruct image.png
    pypenumbra binary_reconstruct image.std 2140 1760 dtype="uint16"

```

Details about these commands and command flags/options can be found through the use of the --help flag.

## Resources

The original paper by Dr. Giovanni Di Domenico:

```text

    Di Domenico, Giovanni, et al. "X‐ray focal spot reconstruction by circular penumbra analysis—Application to digital radiography systems." Medical physics 43.1 (2016): 294-302.

```

The original ImageJ plugin that this code is based off of:

http://www.fe.infn.it/~didomeni/focalspot/focalspot.html
