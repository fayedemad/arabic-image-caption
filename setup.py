from setuptools import setup, find_packages

setup(
    name="anlp",
    version="0.1",
    packages=find_packages(),
    package_dir={"": "."}  # This tells setuptools to look in the current directory
)