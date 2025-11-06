# Installation

The code is designed for developers. To install, first clone the repository and then create a conda environment called vrAnalysis. You can use the `environment.yml` file to create the environment, but I'd actually recommend *not* doing this and instead making an empty environment (python >= 3.9!!!!) and then installing the package locally with `pip install -e .`. 

## Basic Code Installation

1. Clone the repository:

```bash
git clone https://github.com/landoskape/vrAnalysis
cd vrAnalysis
```

2. Create a conda environment called vrAnalysis:

```bash
conda env create -n vrAnalysis
conda activate vrAnalysis
```

3. Install the package in development mode:

```bash
pip install -e .
```

Note: there are "extra" dependencies that will not be installed by this method. The options are:

- registration: includes packages that require special compilers
    - oasis-deconv
    - cvxpy
- gui: includes packages that are used for the GUI for manually adding data to the database
    - pyqt5
    - pyqtgraph
    - napari[all]
- all: includes all of the above

To install the extra dependencies, use one of the following commands:

```bash
pip install -e .[registration]
pip install -e .[gui]
pip install -e .[all]
```

## Other dependencies

As an analysis package of experimental data, there are a few other things that need to happen before you can actually use the package. They primarily relate to setting up your database, your data directories, and some other preprocessing that needs to happen with other python packages (primarily `suite2p` and `ROICaT`).

For a detailed guide on all of this, see the [registration workflow](workflows/registration.md) page which walks you through the whole process. 


## PyTorch

Some of the package uses pytorch. For reasons that escape me, meta has not found a good way to install pytorch inside other packages using `pip install` or `conda install`. My workaround is to make my environment as described above, install `vrAnalysis`, then install pytorch manually using the instructions on the [pytorch website](https://pytorch.org/get-started/locally/).

If you try to use things that depend on pytorch and haven't done this, you'll get import errors. Sorry. 

!!! tip
    Make sure to install PyTorch with the correct CUDA version for your GPU, and verify that GPU acceleration is working properly after installation.

## Configuration

After installation, you have to configure paths for your data:

1. **Data Directory**: Set your local data path in `vrAnalysis/files.py`:

```python
def local_data_path() -> Path:
    return Path("C:/path/to/your/data")
```

2. **Database Paths**: Configure database paths in `vrAnalysis/database.py` using the `get_database_metadata()` function.

