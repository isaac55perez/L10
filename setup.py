from setuptools import setup, find_packages

setup(
    name="random-binary-generator",
    version="1.0.0",
    description="Generate random binary data with specific probability distributions",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.2.0",
    ],
    python_requires=">=3.6",
)