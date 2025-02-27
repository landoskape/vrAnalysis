"""
This module contains support for using ROICaT. It primarily focuses on the
pipeline that uses ROINeT to classify ROIs into different categories (including
cells, dendrites, etc.). 

Note that although this module is in the vrAnalysis repo, it requires a distinct
environment with ROICaT installed to use. To set this up, follow these steps:

1. Create a new environment
2. Install ROICaT
3. Install vrAnalysis in the same environment without dependencies (some aren't compatible with ROICaT)
4. Install the remaining dependencies manually that are needed for vrAnalysis 
   - I didn't write down which ones we need, so just try to run it and see what fails. 

```bash
mamba create -n roicat python=3.11
pip install roicat[all]
pip install vrAnalysis --no-deps
pip install any other dependencies that are needed...
```
"""

from .files import get_classifier_files, get_results_path
