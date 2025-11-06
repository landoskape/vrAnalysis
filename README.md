# vrAnalysis

Tools for processing and analyzing data from behavior and imaging experiments.
The package contains processing methods for data that is specific to the 
experiments I'm doing, but can be rewritten to work with any experimental 
system, there's a few places where user-specific functions need to be 
re-written (which are discussed in the documentation). 

The experimental backbone is [Rigbox](https://github.com/cortex-lab/Rigbox), 
which uses an experiment management program called "Timeline". Behavioral
processing is designed to work with experiments operated by the
[vrControl](https://github.com/landoskape/vrControl) package in conjunction
with timeline. These are virtual reality experiments where subjects (mice) run
on a virtual linear track to collect rewards. The imaging processing is designed 
to work with the standard output of [suite2p](https://github.com/MouseLand/suite2p).

Finally, the package assumes [Alyx](https://github.com/cortex-lab/alyx) 
database structure. This means that the data should be organized in a 
directory first by mouse, then by date, then by session ID. These files should
be contained in a directory, which you can set in the
[files](https://github.com/landoskape/vrAnalysis/blob/main/vrAnalysis/files.py#L7)
module with the `local_data_path` method.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/landoskape/vrAnalysis
cd vrAnalysis
```

2. Create a conda environment (Python >= 3.9) and install in development mode:
```bash
conda create -n vrAnalysis python>=3.9
conda activate vrAnalysis
pip install -e .
```

3. Optional: Install extra dependencies:
```bash
pip install -e .[registration]  # for deconvolution and registration
pip install -e .[gui]           # for GUI tools
pip install -e .[all]           # includes everything
```

4. Install PyTorch separately following the [official instructions](https://pytorch.org/get-started/locally/). Not necessary for most components of the package (more so the "dimilibi" stuff).

5. Configure paths: Set your data directory in `vrAnalysis/files.py` and database paths in `vrAnalysis/database.py`.

For detailed setup instructions including database configuration and preprocessing workflows, see the [installation guide](https://landoskape.github.io/vrAnalysis/installation/) and [registration workflow](https://landoskape.github.io/vrAnalysis/workflows/registration/).


## Documentation
This is not professional software, but it's hopefully useful to people (especially maybe 1 other person in cortex lab). So anyway I made a "proper" documentation system which you can find [here](https://landoskape.github.io/vrAnalysis/).

Some old documentation that is all outdated but still a tiny bit useful is here:
1. [Standard Workflows](_old_docs_/workflows.md)
2. Components
   - [Database](_old_docs_/database.md)
   - [Red Cell Curation Interface](_old_docs_/redCellGUI.md)
   - [Analyzing Data](_old_docs_/analysis.md)
