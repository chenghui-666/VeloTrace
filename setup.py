from setuptools import setup, find_packages

setup(
    name="velotrace",
    version="0.1.0",
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
