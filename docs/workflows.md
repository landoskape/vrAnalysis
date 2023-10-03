# vrAnalysis: workflows

This is documentation for standard workflows using the vrAnalysis package. The
main goal of this file is to remind me what I've written so I don't forget. 
There may be room for explaining what is going on, but mostly it's just code 
blocks indicating standard usage. 


## Red Cell Quality Control Workflow
The red cell QC control workflow is pretty simple. It requires the following 
import statements and the database object:
```python
from vrAnalysis import session
from vrAnalysis import database
from vrAnalysis.redgui import redCellGUI as rgui
vrdb = database.vrDatabase()
```

1. Identify which sessions need red cell QC
The database can create an iterable list of sessions that require red cell 
quality control, along with any other filters. I usually start by iterating
through this list and printing the session names. Here, I'm filtering to only
include sessions from one mouse, to focus curation on a single brain. 
```
for ses in vrdb.iterSessionRedCell(mouseName='ATL027'):
    print(ses.sessionPrint())
```

2. Open up a red cell GUI viewer for one of the sessions
Then, choose a session to curate, and open up the red cell GUI viewer. Note 
that I've only tested this in Jupyter lab. The following code block will open
up a GUI based on PyQt5 and Napari. 
```python
mouseName = 'ATL027'
dateString = '2023-08-04'
sessionid = '701'

# Load registered vrExperiment
vrexp = session.vrExperiment(mouseName, dateString, sessionid)
redCell = session.redCellProcessing(vrexp)
redSelection = rgui.redSelectionGUI(redCell)
```

3. Do curation in the GUI
This all happens in the GUI. 

4. Save your work as one files


5. Tell the database that you are finished curating

6. Check your work
You can also print a list of sessions that have finished quality control. I 
like doing this to sanity check that I curated successfully (where 
successfully means that I pressed the save and update DB button). 
```python
for ses in vrdb.iterSessions(mouseName='ATL027', redCellQC=True):
    print(ses.sessionPrint())
```
