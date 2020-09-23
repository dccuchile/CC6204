import re

import setuptools

VERSIONFILE = "cc6204/_version.py"
verstrline = open(VERSIONFILE, "r").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError(f"Unable to find version string in {VERSIONFILE}.")

long_description = open("README.md", "r").read()

setuptools.setup(
    name="cc6204",
    version=verstr,
    author="Juan-Pablo Silva",
    author_email="jpsilva@dcc.uchile.cl",
    description="Basic autocorrector for CC6204",
    long_description=long_description,
    url="https://github.com/dccuchile/CC6204/tree/master/autocorrector",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "requests",
        "numpy",
        "torch"
    ]
)
