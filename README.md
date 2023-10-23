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
on a virtual linear track to collect rewards. 

The imaging processing is designed to work with the standard output of 
suite2p, so if you are looking for a system to help analyze 2P data then 
you'll only need to change a few things. 

Finally, the package assumes [Alyx](https://github.com/cortex-lab/alyx) 
database structure. 

## Installation
There are two ways to install the vrAnalysis package. The first is by working
directly from the code after cloning the repository from GitHub. First, 
navigate to whatever folder you want to store the cloned repository in. Then, 
clone the repository, make a new conda (or mamba!) environment, and install 
the requirements via the `environment.yml` file.
```
git clone https://github.com/landoskape/vrAnalysis
cd vrAnalysis
conda env create -f environment.yml # try mamba instead of conda if you haven't yet
```

Alternatively, you can do a pip install from GitHub if you don't want to edit 
the code at all. For this method, note that you have to choose which features
you want to include. The core requirements exclude some packages required for 
reprocessing deconvolved calcium traces and exclude the red cell GUI. 

In an existing conda environment, type:
```
pip install git+https://github.com/landoskape/vrAnalysis # core requirements
```

Or, for the extra installs, use one of the following. The registration 
component includes some packages that require special compilers. The gui
component includes some plotting and interactivity related packages that I've
found can conflict with other packages if you're using jupyter. 
```
# pip install git+https://github.com/landoskape/vrAnalysis[registration] # include the registration packages
# pip install git+https://github.com/landoskape/vrAnalysis[gui] # include the red cell GUI
# pip install git+https://github.com/landoskape/vrAnalysis[all] # include everything
```

## Capabilities
vrAnalysis can do the following things: 
- Manage a database (can work with any SQL database with a few adjustments). 
- Register and preprocess experimental data including behavior and imaging
  (will include facecam data soon).  
- Manually annotate results, especially related to whether cells express
  a counter-fluorophore (finished) and for annotating tracked ROIs (not coded
  yet). 
- Analyzing data using organized analysis objects that make it easy to do an
  analysis across groups of sessions from the database.
  
## Documentation
This isn't even close to a professional package so the documentation is 
primarily oriented towards helping future me remember what I've written. To
that end, the documentation files contain a commentary on how to use the 
repository with lots of code blocks indicating standard usage. So, hopefully
that's useful to anyone trying to use the repository or at least just to see
what using this repository will look like.

Documentation Pages
1. [Standard Workflows](docs/workflows.md)
2. Components
   - [Database](docs/database.md)
   - [Red Cell Curation Interface](docs/redCellGUI.md)
   - [Analyzing Data](docs/analysis.md)
