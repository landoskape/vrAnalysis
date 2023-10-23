# vrAnalysis Documentation: Red Cell GUI and Curation

This file explains how the red cell selection gui works. The code for the GUI
can be found [here](../vrAnalysis/redgui/redCellGUI.py) if you want more 
details than are provided by this documentation file.

See the [README](..) for details on how to install the GUI. Note that if you 
installed via pip, you have to include the `[gui]` extras in the install. Once
it's installed, you can import the gui with a standard package import line:
```python
from vrAnalysis.redgui import redCellGUI as rgui
```

At the moment, this GUI is only compatible with data formatted and stored as
`vrExperiment` session objects. However, even though the GUI makes use of the
vrExperiment functionality, the core requirements are pretty simple, so if you
find yourself wanting to use this GUI and don't want to reprocess your data, 
let me know and I can rewrite a loading package for "raw" inputs. 

## Input to the GUI
The GUI loads required variables from the `redCellProcessing` object that you
can find in the [`session`](../vrAnalysis/session.py) module. The GUIs 
extended functionality depends on many features of the `redCellProcessing` 
object, so packaging your data as `vrExperiment` objects and then loading into
the GUI this way is recommended. That being said, the core requirements of the
GUI are pretty simple so there may be an alternative (contact me for support).

### Opening the GUI
In the standard method, the GUI is opened with a `redCellProcessing` object, 
which is created from a `vrExperiment` object (the core processing object of
this repository). The following code block contains an example of what that 
looks like.
```python
# choose a session by picking the mouse name, datestring, and session ID.
mouseName = 'ATL027'
dateString = '2023-08-01'
sessionid = '701'

# Load registered vrExperiment
vrexp = session.vrExperiment(mouseName, dateString, sessionid)
redCell = session.redCellProcessing(vrexp) # create red cell processing object
redSelection = rgui.redSelectionGUI(redCell) # open the GUI
```

### Core input requirements
The `redCellProcessing` object provides a few variables and methods that the 
GUI needs to open and operate. 

#### Variables
1. Number of planes
2. Number of ROIs per plane
3. The stack of reference images containing the red fluorescence data
4. The data describing the ROI masks in a session
5. A list of feature names used to classify cells
6. The plane index associated with each ROI

#### Methods
1. A method of loading the pre-computed features for each ROI
   - note that the feature arrays stored as .npy files are hard-coded right
     now, but in principle it would be easy to make this customizable for
     different features with their own names.
   - The features required are:
     - 'mpciROIs.redS2P'
     - 'mpciROIs.redDotProduct'
     - 'mpciROIs.redPearson'
     - 'mpciROIs.redPhaseCorrelation'
     - 'mpciROIs.redCellManualAssignments'
2. A method for saving the data in the right folder
3. A method for naming the save strings based on the feature name
4. A method for printing the name of the sesssion

## Using the GUI
:)

   






