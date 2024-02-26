from setuptools import setup, find_packages

with open("README.md", mode="r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

setup(
    name="OG2021",
    version="0.1.0",
    author="Erik Novak, Erik Calcina",
    author_email="erik.novak@ijs.si, erik.calcina@ijs.si",
    description="Creating the OG2021 dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[req for req in requirements if req[:2] != "# "],
    setup_requires=["flake8"],
)
