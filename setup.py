from setuptools import setup, find_packages

setup(
    name="velotrace",
    version="0.1.0",
    #description="A small package extracted from a notebook for pseudotime/trajectory utilities.",
    #author="Your Name",
    #author_email="your_email@example.com",
    packages=find_packages(),
    install_requires=[
        "anndata",
        "matplotlib",
        "numpy",
        "pandas",
        "scanpy",
        "sctour",
        "scvelo",
        "scikit-learn",
        "scipy",
        "torch",
        "torchdiffeq",
    ],
    python_requires=">=3.8",
)
