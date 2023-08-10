import pyodbc
from contextlib import contextmanager
import pandas as pd
from pandasgui import show
import vrExperiment as vre

def vrDatabasePath(dbName):
    dbdict = {
        'vrDatabase':r'C:\Users\andrew\Documents\localData\vrDatabaseManagement\vrDatabase.accdb'
    }
    if dbName not in dbdict.keys(): raise ValueError(f"Did not recognize database={dbName}, valid database names are: {[key for key in dbdict.keys()]}")
    return dbdict[dbName]
    

class vrDatabase:
    def __init__(self):
        self.tableName = 'sessiondb'
        self.dbName = 'vrDatabase'
        self.dbpath = vrDatabasePath(self.dbName)
        
    def connect(self):
        connString = (
            r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
            fr"DBQ={self.dbpath};"
        )
        return pyodbc.connect(connString)
    
    @contextmanager
    def openCursor(self, commitChanges=False):
        try:
            # Attempt to open a cursor to the database
            conn = self.connect()
            cursor = conn.cursor()
            yield cursor
        except Exception as ex:
            print(f"An exception occurred while trying to connect to {self.dbName}!")
            raise ex
        else:
            # if no exception was raised, commit changes
            if commitChanges: conn.commit()
        finally:
            # Always close the cursor and connection
            cursor.close()
            conn.close()
            
    def fieldNames(self):
        with self.openCursor() as cursor:
            fieldNames = [col.column_name for col in cursor.columns(table=self.tableName)]
        return fieldNames
            
    def tableData(self):
        with self.openCursor() as cursor:
            cursor.execute(f"SELECT * FROM {self.tableName}")
            tableElements = cursor.fetchall()
        return tableElements
    
    def getTable(self):
        fieldNames = self.fieldNames()
        tableData = self.tableData()
        return pd.DataFrame.from_records(tableData, columns=fieldNames)
        
    def showTable(self, table=None):
        show(self.getTable() if table is None else table)
    
    def needsRegistration(self): 
        df = self.getTable()
        return df[df['vrRegistration']==False]
    
    def needsS2P(self):
        df = self.getTable()
        return df[(df['Imaging']==True) & (df['suite2p']==False)]
    
    def printRequiresS2P(self):
        need = self.needsS2P()
        for idx, row in need.iterrows():
            print(f"Database indicates that suite2p has not been run: {self.vrSession(row).sessionPrint()}")

    def checkS2P(self, withDatabaseUpdate=False, returnCheck=False):
        df = self.getTable()
        
        # return dataframe of sessions with imaging where suite2p wasn't done, even though the database thinks it was
        check_s2pDone = df[(df['Imaging']==True) & (df['suite2p']==True)]
        checked_notDone = check_s2pDone.apply(lambda row: not(self.vrSession(row).suite2pPath().exists()), axis=1)
        notActuallyDone = check_s2pDone[checked_notDone]
        
        # return dataframe of sessions with imaging where suite2p was done, even though the database thinks it wasn't
        check_s2pNeed = df[(df['Imaging']==True) & (df['suite2p']==False)]
        checked_notNeed = check_s2pNeed.apply(lambda row: self.vrSession(row).suite2pPath().exists(), axis=1)
        notActuallyNeed = check_s2pNeed[checked_notNeed]
        
        # Print database errors to workspace
        for idx, row in notActuallyDone.iterrows(): print(f"Database said suite2p has been ran, but it actually hasn't: {self.vrSession(row).sessionPrint()}")
        for idx, row in notActuallyNeed.iterrows(): print(f"Database said suite2p didn't run, but it already did: {self.vrSession(row).sessionPrint()}")
        
        # If withDatabaseUpdate==True, then correct the database
        if withDatabaseUpdate: 
            for idx, row in notActuallyDone.iterrows():
                cUID = row['Unique Session ID']
                updateStatement = f"UPDATE {self.tableName} SET suite2p = False WHERE [Unique Session ID] = {cUID};"
                with self.openCursorCommit(commitChanges=True) as cursor:
                    cursor.execute(updateStatement)
                    
            for idx, row in notActuallyNeed.iterrows():
                cUID = row['Unique Session ID']
                updateStatement = f"UPDATE {self.tableName} SET suite2p = True WHERE [Unique Session ID] = {cUID};"
                with self.openCursor(commitChanges=True) as cursor:
                    cursor.execute(updateStatement)
                    
        # If returnCheck is requested, return True if any records were invalid
        if returnCheck: return checked_notDone.any() or checked_notNeed.any()
    
    def sessionName(self, row):
        mouseName = row['Mouse Name']
        sessionDate = row['Session Date'].strftime('%Y-%m-%d')
        sessionID = str(row['Session ID'])
        return mouseName, sessionDate, sessionID

    def vrSession(self, row):
        mouseName, sessionDate, sessionID = self.sessionName(row)
        return vre.vrSession(mouseName, sessionDate, sessionID)
    
    def vrRegistration(self, row):
        mouseName, sessionDate, sessionID = self.sessionName(row)
        return vre.vrExperimentRegistration(mouseName, sessionDate, sessionID)
    
    def registerSession(self, **userOpts):
        # Options for data management:
        # These are a subset of what is available in vre.vrExperimentRegistration 
        # They indicate what preprocessing steps to take depending on what was performed in each experiment
        opts = {}
        opts['vrBehaviorVersion'] = 1 # 1==standard behavioral output (will make conditional loading systems for alternative versions...)
        opts['facecam'] = False # whether or not face video was performed on this session (note: only set to True when DLC has already been run!)
        opts['imaging'] = True # whether or not imaging was performed on this session (note: only set to True when suite2p has already been run!)
        opts['oasis'] = True # whether or not to rerun oasis on calcium signals (note: only used if imaging is run)
        opts['moveRawData'] = False # whether or not to move raw data files to 'rawData'
        opts['redCellProcessing'] = True # whether or not to preprocess redCell features into oneData using the redCellProcessing object (only runs if redcell in self.value['available'])
        
        assert userOpts.keys() <= opts.keys(), f"userOpts contains the following invalid keys: {set(userOpts.keys()).difference(opts.keys())}"
        opts.update(userOpts) # Update default opts with user requests
        
        print("In registerSessions, 'vrBehaviorVersion' is an important input that hasn't been coded yet!") 
        
        dfToRegister = self.needsRegistration()
        for idx, row in dfToRegister.iterrows():
            cUID = row['Unique Session ID']
            opts['imaging']=row['Imaging']
            opts['facecam']=row['Face Camera']
            vrExpReg = self.vrRegistration(row, opts)
            try: 
                print(f"Performing vrExperiment preprocessing for session: {vrExpReg.sessionPrint()}")
                vrExpReg.doPreprocessing()
                vrExpReg.saveParams()
            except Exception as ex:
                print(f"The following exception was raised when trying to preprocess session: {vrExpReg.sessionPrint()}")
                print(f"Exception: {ex}")
            else:
                updateStatement = f"UPDATE {self.tableName} SET vrRegistration = True WHERE [Unique Session ID] = {cUID};"
                with self.openCursor(commitChanges=True) as cursor: 
                    cursor.execute(updateStatement)
            finally: 
                del vrExpReg
                    
            
        
# def checkSessionFiles(mouseName, fileIdentifier, onlyTrue=True):
#     dataPath = fm.localDataPath()
#     sessionPaths = glob(str(dataPath / mouseName) + '/*')
#     for spath in sessionPaths:
#         cdate = Path(spath).name
#         sessions = glob(spath+'/*/')
#         for s in sessions:
#             cses = Path(s).name
#             targetFiles = glob(s+fileIdentifier)
#             numFiles = len(targetFiles)
#             if not(onlyTrue and numFiles==0): 
#                 print(f"{cdate}    {cses}    numFiles: {numFiles}")
            
# checkSessionFiles('ATL012', '*eye.mj2') #'*eye.mj2' / 'suite2p')