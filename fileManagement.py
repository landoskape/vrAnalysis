from pathlib import Path
import pandas as pd
import pyarrow as pa

def codePath():
    return Path('C:/Users/andrew/Documents/GitHub/vrAnalysis')

def localDataPath():
    return Path("C:/Users/andrew/Documents/localData")

def serverPath():
    return Path("//zaru.cortexlab.net/Subjects")

def getCopyString(mouseName, datestr='', session='', server=serverPath()):
    sourceString = Path(server / mouseName / datestr / session)
    targetString = Path(localDataPath() / mouseName / datestr / session)
    cmdPromptCommand = f"robocopy {sourceString} {targetString} /s /xf *.tif"
    print(cmdPromptCommand)




