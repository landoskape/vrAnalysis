# With help from: https://github.com/pypa/sampleproject

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

deps_core = [
    "numpy",
    "scipy",
    "numba",
    "matplotlib",
    "jupyterlab",
    "oasis-deconv",
    "cvxpy",
    "tqdm",
    "pyodbc",
    "pandas",
    "numpyencoder",
]
    
deps_registration = [
    "oasis-deconv",
    "cvxpy",
]

deps_gui = [
    "pyqt5",
    "pyqtgraph",
    "napari[all]",
    "vrAnalysis.redgui",
]

setup(
    name="vrAnalysis",
    version="1.0.0",
    description="code to analyze behavior and imaging data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/landoskape/vrAnalysis",
    author="Andrew T. Landau",
    license='LICENSE',
    packages=["vrAnalysis",],
    python_requires=">=3.9, <=4",
    install_requires=deps_core,  
    extras_require={
        "registration": deps_registration, 
        "gui": deps_gui, 
        "all": deps_registration+deps_gui,
    }
)