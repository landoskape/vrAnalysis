# With help from: https://github.com/pypa/sampleproject

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Core code for managing sessions and the database
deps_core = [
    "numpy",
    "scipy",
    "numba",
    "matplotlib",
    "ipympl",
    "jupyterlab",
    "tqdm",
    "pyodbc",
    "pandas",
    "numpyencoder",
    "speedystats",
    "joblib",
    "syd",
]

# Extra code for performing registration (these things are extras specifically for oasis)
deps_registration = [
    "oasis-deconv",
    "cvxpy",
]

# Extra packages for managing the redCellQC GUI
# sometimes these can get weird with other plotting and visualization requirements, so I'm keeping them separate
deps_gui = [
    "pyqt5",
    "pyqtgraph",
    "napari[all]",
]

setup(
    name="vrAnalysis",
    version="2.0.0",
    description="code to analyze behavior and imaging data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/landoskape/vrAnalysis",
    author="Andrew T. Landau",
    license="LICENSE",
    packages=find_packages(),
    python_requires=">=3.9, <=4",
    install_requires=deps_core,
    extras_require={
        "registration": deps_registration,
        "gui": deps_gui,
        "all": deps_registration + deps_gui,
    },
)
