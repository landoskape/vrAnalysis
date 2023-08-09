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
        self.readOnly = readOnly
    
    def connect(self, readOnly=True):
        connString = (
            r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
            fr"DBQ={self.dbpath};"
        )
        return pyodbc.connect(connString, readonly=readOnly)
    
    @contextmanager
    def openCursor(self, commitChanges=False):
        try:
            conn = self.connect()
            cursor = conn.cursor()
            yield cursor
        except Exception as e:
            print(f"An exception occurred while trying to connect to {self.dbName}!")
            raise e
        finally:
            if commitChanges: conn.commit()
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
        
    def showTable(self):
        show(self.getTable())
    
    def needsS2P(self):
        df = self.getTable()
        return df[(df['Imaging']==True) & (df['suite2p']==False)]
    
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
    
    def vrSession(self, row):
        return vre.vrSession(row['Mouse Name'], row['Session Date'].strftime('%Y-%m-%d'), str(row['Session ID']))
        
        