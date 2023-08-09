import pyodbc

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
    
    def close(self,conn):
        conn.close()
        
    def printValues(self):
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {self.tableName}")
        for row in cursor.fetchall():
            print(row)
        cursor.close()
        conn.close()
