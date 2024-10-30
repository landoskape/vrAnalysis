from pathlib import Path
from tkinter import Tk


def codePath():
    return Path("C:/Users/andrew/Documents/GitHub/vrAnalysis")


def localDataPath():
    return Path("C:/Users/andrew/Documents/localData")


def literatureDataPath():
    return Path("C:/Users/andrew/Documents/literatureData")


def storagePath():
    return Path("D:/localData")


def sharedDataPath():
    return localDataPath() / "sharedData"


def analysisPath():
    return localDataPath() / "analysis"


def figurePath():
    return localDataPath() / "_figure_library"


def serverPath(zaru=True):
    name = "zaru" if zaru else "zortex"
    return Path(f"//{name}.cortexlab.net/Subjects")


def getCopyString(mouseName, datestr="", session="", server=serverPath(), toClipboard=True):
    sourceString = Path(server / mouseName / datestr / session)
    targetString = Path(localDataPath() / mouseName / datestr / session)
    cmdPromptCommand = f"robocopy {sourceString} {targetString} /s /xf *.tif *.mj2"
    if toClipboard:
        tkManager = Tk()
        tkManager.clipboard_append(cmdPromptCommand)
        tkManager.destroy()
    print(cmdPromptCommand)


def copyDataToStorage(mouseName, datestr="", session="", toClipboard=True):
    sourceString = Path(localDataPath() / mouseName / datestr / session)
    targetString = Path(storagePath() / mouseName / datestr / session)
    cmdPromptCommand = f"robocopy {sourceString} {targetString} /s"
    if toClipboard:
        tkManager = Tk()
        tkManager.clipboard_append(cmdPromptCommand)
        tkManager.destroy()
    print(cmdPromptCommand)


def s2pTargets(*inputs, zaru=True):
    if len(inputs) == 3:
        mouseName, dateString, session = inputs
    else:
        mouseName = inputs[0].mouseName
        dateString = inputs[0].dateString
        session = inputs[0].session

    sourceString = Path(serverPath(zaru=zaru) / mouseName / dateString / session)
    targetString = Path(localDataPath() / mouseName / dateString / session)
    print(str(sourceString))
    print(str(targetString))


def checkSessionFiles(mouseName, fileIdentifier):
    return None
