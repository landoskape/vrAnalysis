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
    def __init__(self, tableName='sessiondb', dbName='vrDatabase', readOnly=True):
        self.tableName = tableName
        self.dbName = dbName
        self.dbpath = vrDatabasePath(dbName)
        self.readOnly = readOnly
    
    def connect(self):
        connString = (
            r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
            fr"DBQ={self.dbpath};"
        )
        return pyodbc.connect(connString, readonly=self.readOnly)
    
    @contextmanager
    def openCursor(self):
        try:
            conn = self.connect()
            cursor = conn.cursor()
            yield cursor
        except Exception as e:
            print(f"An exception occurred while trying to connect to {self.dbName}!")
            raise e
        finally:
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
        check_s2pDone = df[(df['Imaging']==True) & (df['suite2p']==True)]
        checked_notDone = check_s2pDone.apply(lambda row: not(self.vrSession(row).suite2pPath().exists()), axis=1)
        notActuallyDone = check_s2pDone[checked_notDone]
        
        check_s2pNeed = df[(df['Imaging']==True) & (df['suite2p']==False)]
        checked_notNeed = check_s2pNeed.apply(lambda row: self.vrSession(row).suite2pPath().exists(), axis=1)
        notActuallyNeed = check_s2pNeed[checked_notNeed]
        
        for idx, row in notActuallyDone.iterrows(): print(f"Database said suite2p has been ran, but it actually hasn't: {self.vrSession(row).sessionPrint()}")
        for idx, row in notActuallyNeed.iterrows(): print(f"Database said suite2p didn't run, but it already did: {self.vrSession(row).sessionPrint()}")
        
        if returnCheck: return checked_notDone.any() or checked_notNeed.any()
    
    def vrSession(self, row):
        return vre.vrSession(row['Mouse Name'], row['Session Date'].strftime('%Y-%m-%d'), str(row['Session ID']))
        
        