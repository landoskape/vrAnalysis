# vrAnalysis Documentation: workflows

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

There are two general workflows for curation of red cell assignment. The first
is to go through each session and do the curation independently of all other 
sessions. I refer to this as the "Primary Worklow", because it's pretty 
efficient and I think I'll probably use it this way. Alternatively, you can do
the curation for one or a few sessions and apply it to other sessions, usually
from the same mouse. 

### Primary Workflow

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

3. Do curation in the GUI, save work and update database
This all happens in the GUI. For information on how to use the GUI (it's a 
little tricky to use but powerful for my needs), go over to the
[redCellGUI](redCellGUI.md) documentation page. 

4. Check your work!
You can also print a list of sessions that have finished quality control. I 
like doing this to sanity check that I curated successfully (where 
successfully means that I pressed the save and the update database button). 
Note that this _only_ checks if you updated the database, saving the results 
is a different part of the GUI. (But of course you can always reopen the 
`redCellGUI` for the same session and make sure it looks good). 
```python
for ses in vrdb.iterSessions(mouseName='ATL027', redCellQC=True):
    print(ses.sessionPrint())
```

### Additional Features
Sometimes you'll want to compare the cutoffs you chose for sessions in the 
same (or different) mice. To do so, use this block. Make a session iterable to
go through target list of sessions, generally with a filter on the mouseName 
and only looking at sessions where you've done red cell QC. It'll print a 
pandas dataframe showing the min/max cutoffs for each feature value used in 
the red cell detection. 
```python
rgui.compareFeatureCutoffs(*vrdb.iterSessions(mouseName='ATL027', redCellQC=True), roundValue=3)
```




