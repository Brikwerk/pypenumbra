import io

from setuptools import find_packages
from setuptools import setup

with io.open("README.rst", "rt", encoding="utf8") as f:
    readme = f.read()

setup(
    name="pypenumbra",
    version=0.1,
    packages=['pypenumbra'],
    license="MIT",
    author="Reece Walsh",
    author_email="reece@brikwerk.com",
    description="Computes the focal spot and sinogram from a penumbra image.",
    long_description=readme,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    include_package_data=True,
    python_requires="!=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*",
    install_requires=[
        "numpy>=1.17.3",
        "opencv-python>=4.1.1.26",
        "scikit-image>=0.16.2",
        "fire>=0.2.1",
    ],
    extras_require={
        "dev": [
            "pytest",
            "coverage",
            "tox",
        ],
    },
    entry_points={
        "console_scripts": [
            "pypenumbra = pypenumbra.cli:main"
        ]
    },
)