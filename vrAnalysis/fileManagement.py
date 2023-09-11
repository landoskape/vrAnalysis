from pathlib import Path
from tkinter import Tk

def hello():
    print('hello world')
    
def codePath():
    return Path('C:/Users/andrew/Documents/GitHub/vrAnalysis')

def localDataPath():
    return Path("C:/Users/andrew/Documents/localData")

def analysisPath():
    return localDataPath() / 'analysis'
    
def serverPath():
    return Path("//zaru.cortexlab.net/Subjects")

def getCopyString(mouseName, datestr='', session='', server=serverPath(), toClipboard=True):
    sourceString = Path(server / mouseName / datestr / session)
    targetString = Path(localDataPath() / mouseName / datestr / session)
    cmdPromptCommand = f"robocopy {sourceString} {targetString} /s /xf *.tif *.mj2" 
    if toClipboard:
        tkManager = Tk()
        tkManager.clipboard_append(cmdPromptCommand)
        tkManager.destroy()
    print(cmdPromptCommand)

def s2pTargets(mouseName, dateString='', session='', server=serverPath()):
    sourceString = Path(serverPath() / mouseName / dateString / session)
    targetString = Path(localDataPath() / mouseName / dateString / session)
    print(sourceString)
    print(targetString)

def checkSessionFiles(mouseName, fileIdentifier):
    return None



