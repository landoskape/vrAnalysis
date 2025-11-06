# vrAnalysis Documentation: workflows

This is documentation for standard workflows using the vrAnalysis package. The
main goal of this file is to remind me what I've written so I don't forget. 
There may be room for explaining what is going on, but mostly it's just code 
blocks indicating standard usage. 


## Red Cell Quality Control Workflow
The red cell QC control workflow is pretty simple. It requires the following 
import statements and the database object:
```python
from _old_vrAnalysis import session
from _old_vrAnalysis import database
from _old_vrAnalysis.redgui import redCellGUI as rgui
vrdb = database.vrDatabase()
```

Red cell quality control uses four features computed automatically during 
registration, along with additional manual curation. There is a nice GUI that
you can use to choose thresholds for each feature and to manually toggle 
whether a cell is red or not in addition to the feature criterion. The 
full documentation page for the GUI is [here](redCellGUI.md). 

There are two general workflows for curation of red cell assignment. The first
is to go through each session and do the curation independently of all other 
sessions. I refer to this as the "Primary Worklow", because it's pretty 
efficient and you always have to start with this. 

The second method is to do the curation for one session and then apply it to 
other sessions automatically, (usually to sessions from the same mouse). This 
is obviously efficient, because it only requires direct curation from one 
session, but I recommend at least checking all sessions, especially because 
you can do a little extra manual curation after choosing the feature 
thresholds. 

### Primary Workflow

1. Identify which sessions need red cell QC
The database can create an iterable list of sessions that require red cell 
quality control, along with any other filters. I usually start by iterating
through this list and printing the session names. Here, I'm filtering to only
include sessions from one mouse, to focus curation on a single brain. 
```python
for ses in vrdb.iterSessionNeedRedCellQC(mouseName='ATL027'):
    print(ses.sessionPrint())
```

2. Open up a red cell GUI viewer for one of the sessions
Then, choose a session to curate, and open up the red cell GUI viewer. Note 
that I've only tested this in Jupyter lab. The following code block will open
up a GUI based on PyQt5 and Napari. 
```python
mouseName = 'ATL027'
dateString = '2023-08-01'
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

### Secondary Workflow
1. Perform the primary workflow for at least one session, then save.
Once you do this, the threshold values for each feature will be stored in the
one-data for the session you loaded. 

2. Apply the features to other sessions.
Now, you can apply those threshold values you selected to other sessions. 
First, create a `redCellProcessing` object for the session you already 
curated. In the example, I curated session 'ATL027/2023-08-01/701'. Then, go
through all sessions you want to copy the thresholds to with an iterator of 
sessions that need red cell QC (usually from the same mouse). Make a
`redCellProcessing` object for each session, then use the `updateFromSession`
method to copy threshold from one session to another. Note that I'm using 
`autoload=False` because `redCellProcessing` objects load data by default,
which takes a few seconds, and we only need the class methods for this 
purpose.
```python
copyCriterionFrom = session.redCellProcessing('ATL027','2023-08-01','701', autoload=False)
for ses in vrdb.iterSessionNeedRedCellQC(mouseName='ATL027'):
    redCell = session.redCellProcessing(ses, autoload=False)
    redCell.updateFromSession(copyCriterionFrom)
```

### Additional Features
Sometimes you'll want to compare the cutoffs you chose for sessions in the 
same (or different) mice. To do so, use this block. Make a session iterable to
go through a target list of sessions, generally with a filter on the 
`<mouseName>` and only looking at sessions where you've done red cell QC. 
It'll print a pandas dataframe showing the min/max cutoffs for each feature 
value used in the red cell detection. You may want to increase the display 
width of pandas before doing this in whatever notebook you're working in.
```python
import pandas as pd
pd.options.display.width = 1000
rgui.compareFeatureCutoffs(*vrdb.iterSessions(mouseName='ATL027', redCellQC=True), roundValue=3)
```

Note that if you use the secondary workflow, then the feature cutoffs should 
be the same for every session. It's a good way to check your work!




