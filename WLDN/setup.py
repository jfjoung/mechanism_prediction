import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wlnfw",
    version="2.0",
    author="Thijs Stuyver, Thomas Struble",
    author_email="tstuyver@mit.edu, thomasstruble@gmail.com",
    description="Reaction forward prediction training pipeline code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/mlpds_mit/ASKCOS/wln-keras-fw",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
