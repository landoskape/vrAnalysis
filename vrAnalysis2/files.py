from pathlib import Path
from tkinter import Tk


def repo_path() -> Path:
    return Path("C:/Users/Andrew/Documents/GitHub/vrAnalysis")


def local_data_path() -> Path:
    # return Path("C:/Users/andrew/Documents/localData")
    return storage_path()


def literature_data_path() -> Path:
    return Path("C:/Users/andrew/Documents/literatureData")


def storage_path() -> Path:
    return Path("D:/localData")


def analysis_path() -> Path:
    return local_data_path() / "analysis"


def server_path(server: str = "zortex") -> Path:
    return Path(f"//{server}.cortexlab.net/Subjects")


def getCopyString(mouseName, datestr="", session="", server=server_path(), toClipboard=True):
    sourceString = Path(server / mouseName / datestr / session)
    targetString = Path(local_data_path() / mouseName / datestr / session)
    cmdPromptCommand = f"robocopy {sourceString} {targetString} /s /xf *.tif *.mj2"
    if toClipboard:
        tkManager = Tk()
        tkManager.clipboard_append(cmdPromptCommand)
        tkManager.destroy()
    print(cmdPromptCommand)


def copyDataToStorage(mouseName, datestr="", session="", toClipboard=True):
    sourceString = Path(local_data_path() / mouseName / datestr / session)
    targetString = Path(storage_path() / mouseName / datestr / session)
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

    sourceString = Path(server_path(zaru=zaru) / mouseName / dateString / session)
    targetString = Path(local_data_path() / mouseName / dateString / session)
    print(str(sourceString))
    print(str(targetString))
