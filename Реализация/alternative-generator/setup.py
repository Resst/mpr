from setuptools import setup, find_packages

setup(
    name="alternative-generator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
    ],
    python_requires=">=3.7",
    author="Lebedev Vladimir",
    description="A package for generating alternatives.",
)