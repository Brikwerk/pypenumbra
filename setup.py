import io

from setuptools import find_packages
from setuptools import setup

with io.open("README.rst", "rt", encoding="utf8") as f:
    readme = f.read()

setup(
    name="Flask",
    version=0.1,
    license="MIT",
    author="Reece Walsh",
    author_email="reece@brikwerk.com",
    description="Computes the focal spot from a penumbra image.",
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
    packages=find_packages(""),
    package_dir={"": "pypenumbra"},
    include_package_data=True,
    python_requires="!=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*",
    install_requires=[],
    extras_require={
        "dev": [
            "pytest",
            "coverage",
            "tox",
            "sphinx",
        ],
    },
    entry_points={"console_scripts": ["pypenumbra = pypenumbra.cli:main"]},
)