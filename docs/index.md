# vrAnalysis Documentation

Welcome to the documentation for **vrAnalysis**, a comprehensive Python package for processing and analyzing virtual reality (VR) behavioral and imaging experiments.

## Overview

vrAnalysis is designed to work with:
- **Behavioral data** from [vrControl](https://github.com/landoskape/vrControl) experiments
- **Imaging data** processed with [suite2p](https://github.com/MouseLand/suite2p)
- **Database management** using Microsoft Access databases (or other SQL databases with minimal modifications)
- **Session tracking** following the [Alyx](https://github.com/cortex-lab/alyx) directory structure

## Key Features

- **Database Management**: Track and manage VR session data with an easy-to-use database interface
- **Session Registration**: Automated preprocessing of behavioral and imaging data
- **Data Processing**: Generate spike maps, occupancy maps, and other spatial representations
- **Cell Tracking**: Track cells across sessions for longitudinal analysis
- **Multi-session Analysis**: Analyze data across groups of sessions
- **Quality Control**: Built-in tools for session quality control and annotation

## Quick Navigation

- **[Installation Guide](installation.md)** - Get started with vrAnalysis
- **[Quickstart Tutorial](quickstart.md)** - Learn the basics
- **[Module Documentation](modules/database.md)** - Detailed module descriptions
- **[API Reference](api/vrAnalysis.md)** - Complete API documentation

## Package Structure

vrAnalysis is organized into several main modules:

- **`database`**: Database management and session tracking
- **`sessions`**: Session data loading and management
- **`registration`**: Data preprocessing and registration workflows
- **`processors`**: Data processing pipelines (e.g., spike maps)
- **`analysis`**: Analysis tools and utilities
- **`tracking`**: Cell tracking across sessions
- **`multisession`**: Multi-session analysis capabilities
- **`helpers`**: Utility functions and helpers

## Getting Help

If you encounter issues or have questions:

1. Check the [Quickstart Guide](quickstart.md) for common workflows
2. Review the [Module Documentation](modules/database.md) for specific features
3. Consult the [API Reference](api/vrAnalysis.md) for detailed function signatures

