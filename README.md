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

## Capabilities
vrAnalysis can do the following things: 
- Manage a database (in conjunction with Microsoft Access tables, but can work
  with any SQL database with a few adjustments). 
- Register and preprocess experimental data including behavior, imaging, and 
  potentially facecam data (not coded yet). 
- Manually annotate results, especially related to whether cells express
  a counter-fluorophore and for annotating tracked ROIs (not coded yet). 
- Analyzing data using organized analysis objects that make it easy to do an
  analysis across groups of sessions from the database.
  
## Documentation
This isn't even close to a professional package so the documentation is 
primarily oriented towards helping future me remember what I've written. To
that end, the documentation files contain a commentary on how to use the 
repository with lots of code blocks indicating standard usage. So, hopefully
that's useful to anyone trying to use the repository or at least just to see
what using this repository will look like.

[Documentation Pages](docs)
1. [Standard Workflows](docs/workflows.md)
2. Components
   - [Database](docs/database.md)
   - [Red Cell Curation Interface](docs/redCellGUI.md)
   - [Analyzing Data](docs/analysis.md)
