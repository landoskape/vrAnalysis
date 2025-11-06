# Installation

This guide will help you install vrAnalysis2 and its dependencies.

## Prerequisites

- Python 3.8 or higher (note tested, might be even higher because of typing)
- Conda or Mamba (recommended for managing dependencies, use Mamba for faster installation!)
- Git (for cloning the repository)

## Installation

This method allows you to edit the code and is recommended if you plan to contribute or modify the package.

1. Clone the repository:

```bash
git clone https://github.com/landoskape/vrAnalysis
cd vrAnalysis
```

2. Create a conda environment from the provided `environment.yml`:

```bash
conda env create -f environment.yml
# Or use mamba for faster installation:
mamba env create -f environment.yml
```

3. Activate the environment:

```bash
conda activate vrAnalysis  # or whatever name is specified in environment.yml
```

4. Install the package in development mode:

```bash
pip install -e .
```

## Configuration

After installation, you have to configure paths for your data:

1. **Data Directory**: Set your local data path in `vrAnalysis2/files.py`:

```python
def local_data_path() -> Path:
    return Path("C:/path/to/your/data")
```

2. **Database Paths**: Configure database paths in `vrAnalysis2/database.py` using the `get_database_metadata()` function.

## Verifying Installation

To verify that vrAnalysis2 is installed correctly:

```python
import vrAnalysis
from vrAnalysis.sessions import B2Session, create_b2session
from vrAnalysis.database import get_database_metadata

print("vrAnalysis2 installed successfully!")
```

## Troubleshooting

### Common Issues

1. **Database Connection Errors**: Ensure you have the appropriate database drivers installed (e.g., Microsoft Access ODBC drivers for `.accdb` files).

2. **Import Errors**: Make sure you've activated the correct conda environment and installed all dependencies.

3. **Path Issues**: Verify that your data paths are correctly configured and accessible.

## Next Steps

After installation, check out the [Quickstart Guide](quickstart.md) to begin using vrAnalysis2.

