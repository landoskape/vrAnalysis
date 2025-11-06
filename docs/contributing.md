# Contributing

Thank you for your interest in contributing to vrAnalysis!

## Development Setup

1. Fork and clone the repository
2. Create a conda environment:

```bash
conda env create -f environment.yml
conda activate vrAnalysis
```

3. Install in development mode:

```bash
pip install -e .
```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where possible
- Write docstrings in NumPy format
- Keep line length to 150 characters (per `pyproject.toml`)

## Documentation

- Add docstrings to all public functions and classes
- Use NumPy-style docstrings
- Update relevant documentation pages when adding features
- Include examples in docstrings where helpful

## Testing

- Write tests for new functionality
- Ensure existing tests pass
- Test with real data when possible

## Pull Requests

- Create a new branch for your changes
- Write clear commit messages
- Update documentation as needed
- Ensure all tests pass before submitting

## Questions?

If you have questions about contributing, please open an issue on GitHub.

