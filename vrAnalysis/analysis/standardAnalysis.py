import pickle
import matplotlib.pyplot as plt

from .. import session
from .. import fileManagement as fm


class standardAnalysis:
    """
    Top level class for standard analyses in the vrAnalysis repository.
    """

    def __init__(self, vrexp):
        """
        init function for analysis

        requires a vrExperiment object, which is loaded into the object instance.
        Also names the analysis type, this should be overwritten!!
        """
        self.name = "standardAnalysis"
        self.vrexp = vrexp
        assert type(self.vrexp) == session.vrExperiment, "vrexp is not a vrExperiment object"

    def save_temp_file(self, data, name):
        """
        save a temporary file

        saves a temporary file in the analysis directory with the name provided
        """
        print(f"{self.name} is saving a temporary file for session: {self.vrexp.sessionPrint()}")
        with open(self.saveDirectory("temp") / name, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def clear_temp_files(self):
        """
        clear temporary files

        clears all temporary files in the analysis directory
        """
        print(f"{self.name} is clearing temporary files for session: {self.vrexp.sessionPrint()}")
        for f in (self.saveDirectory("temp")).iterdir():
            f.unlink()

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
        if not (dirName.is_dir()):
            dirName.mkdir(parents=True)
        return dirName

    def saveFigure(self, figNumber, name, extra_name=""):
        """
        save a figure currently open in matplotlib

        attempts to save matplotlib figure(figNumber) in the save directory with a particular name
        """
        print(f"{self.name} is saving a {name} figure for session: {self.vrexp.sessionPrint()}")
        plt.figure(figNumber)
        figpath = self.saveDirectory(name) / str(self.vrexp) / (extra_name + ".png")
        if not figpath.parent.is_dir():
            figpath.parent.mkdir(parents=True)
        plt.savefig(figpath)


class multipleAnalysis(standardAnalysis):
    """
    Class for multi-session analyses in the vrAnalysis repository.
    """

    def __init__(self):
        """
        init function for analysis

        requires a vrExperiment object, which is loaded into the object instance.
        Also names the analysis type, this should be overwritten!!
        """
        self.name = "multipleAnalysis"

    def save_temp_file(self, data, name):
        """
        save a temporary file

        saves a temporary file in the analysis directory with the name provided
        """
        print(f"{self.name} is saving a temporary file:")
        with open(self.saveDirectory("temp") / name, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def saveFigure(self, figNumber, multiname, name):
        """
        save a figure currently open in matplotlib

        attempts to save matplotlib figure(figNumber) in the save directory with a particular name
        """
        print(f"{self.name} is saving a {multiname} figure for {name}")
        plt.figure(figNumber)
        figpath = self.saveDirectory(multiname) / (name + ".png")
        if not figpath.parent.is_dir():
            figpath.parent.mkdir(parents=True)
        plt.savefig(figpath)
