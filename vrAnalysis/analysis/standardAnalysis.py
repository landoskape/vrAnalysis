import matplotlib.pyplot as plt

from .. import session
from .. import fileManagement as fm

class standardAnalysis:
    '''
    Top level class for standard analyses in the vrAnalysis repository.
    '''
    def __init__(self, vrexp):
        self.name = 'standardAnalysis'
        self.vrexp = vrexp
        assert type(self.vrexp)==session.vrExperiment, "vrexp is not a vrExperiment object" 
    
    def analysisDirectory(self):
        return fm.analysisPath()
    
    def saveDirectory(self, name):
        # Define and create target directory
        dirName = self.analysisDirectory() / self.name / name
        if not(dirName.is_dir()): dirName.mkdir(parents=True)
        return dirName
    
    def saveFigure(self, figNumber, name):
        print(f"{self.name} is saving a {name} figure for session: {self.vrexp.sessionPrint()}")
        plt.figure(figNumber)
        plt.savefig(self.saveDirectory(name) / str(self.vrexp))