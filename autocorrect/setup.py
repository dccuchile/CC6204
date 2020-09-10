import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="cc6204",
    version="0.2.0",
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
