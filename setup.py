"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="branch",
    version="0.0",
    description="A PDE solver",
    long_description=long_description,
    url="https://github.com/nguwijy/deep_branching_with_domain",
    author="Nguwi Jiang Yu",
    author_email="nguwijy@hotmail.com",
    packages=find_packages(),
    python_requires=">=3.8, <4",
    install_requires=[
        "matplotlib",
        "numpy",
        "pynverse",
        "ray",
        "scipy",
        "sympy",
        "torch",
    ],
)
