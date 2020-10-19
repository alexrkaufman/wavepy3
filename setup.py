import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wavepy3",
    version="0.0.2",
    author="Alex Kaufman",
    author_email="arkaufman+wavepy3@protonmail.com",
    description="An open source package for wave optics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: ???",
        "Operating System :: OS Independent",
    ],
    python_requires='>= 3.6',
)
