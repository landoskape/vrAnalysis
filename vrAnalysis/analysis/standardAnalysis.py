import matplotlib.pyplot as plt

from .. import session
from .. import fileManagement as fm

class standardAnalysis:
    '''
    Top level class for standard analyses in the vrAnalysis repository.
    '''
    def __init__(self, vrexp):
        """
        init function for analysis
        
        requires a vrExperiment object, which is loaded into the object instance.
        Also names the analysis type, this should be overwritten!!
        """
        self.name = 'standardAnalysis'
        self.vrexp = vrexp
        assert type(self.vrexp)==session.vrExperiment, "vrexp is not a vrExperiment object" 
    
    def analysisDirectory(self):
        """
        return the directory to analysis data
        
        in your `fileManagement` file, hard-code whatever path you want to save analysis data and figures on.
        this method will just use that path.
        """
        return fm.analysisPath()
    
    def saveDirectory(self, name):
        """
        defines an analysis specific save directory
        
        it's always inside the `analysisDirectory()`, but then will add a folder
        for the particular analysis type (i.e. self.name) and then adds an additional
        folder for the specific analysis you are doing (name)
        """
        # Define and create target directory
        dirName = self.analysisDirectory() / self.name / name
        if not(dirName.is_dir()): dirName.mkdir(parents=True)
        return dirName
    
    def saveFigure(self, figNumber, name):
        """
        save a figure currently open in matplotlib
        
        attempts to save matplotlib figure(figNumber) in the save directory with a particular name
        """
        print(f"{self.name} is saving a {name} figure for session: {self.vrexp.sessionPrint()}")
        plt.figure(figNumber)
        plt.savefig(self.saveDirectory(name) / str(self.vrexp))

class multipleAnalysis(standardAnalysis):
    '''
    Class for multi-session analyses in the vrAnalysis repository.
    '''
    def __init__(self):
        """
        init function for analysis
        
        requires a vrExperiment object, which is loaded into the object instance.
        Also names the analysis type, this should be overwritten!!
        """
        self.name = 'multipleAnalysis'
    
    def saveFigure(self, figNumber, multiname, name):
        """
        save a figure currently open in matplotlib
        
        attempts to save matplotlib figure(figNumber) in the save directory with a particular name
        """
        print(f"{self.name} is saving a {multiname} figure for {name}")
        plt.figure(figNumber)
        plt.savefig(self.saveDirectory(multiname) / name)











