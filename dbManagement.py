import pyodbc
from datetime import datetime
from contextlib import contextmanager
import pandas as pd
from pandasgui import show
import vrExperiment as vre
import fileManagement as fm

def vrDatabaseMetadata(dbName):
    """
    Retrieve metadata for a specified database.

    This function retrieves metadata for a specified database from the `dbdict` dictionary.
    The `dbdict` dictionary contains the database paths, names, and primary table name.

    Parameters
    ----------
    dbName : str
        The name of the database for which to retrieve metadata.

    Returns
    -------
    dict
        A dictionary containing metadata for the specified database.
        It has three keys: 'dbPath', 'dbName', and 'tableName'.

    Raises
    ------
    ValueError
        If the provided `dbName` is not recognized as a valid database name.

    Example
    -------
    >>> metadata = vrDatabaseMetadata('vrDatabase')
    >>> print(metadata['dbPath'])
    'C:\\Users\\andrew\\Documents\\localData\\vrDatabaseManagement\\vrDatabase.accdb'
    >>> print(metadata['dbName'])
    'vrDatabase'

    Notes
    -----
    - The `dbdict` dictionary contains metadata for recognized databases.
    - Edit this function to specify the path, database, and primary table on your computer. 
    """
    
    dbdict = {
        'vrDatabase': {
            'dbPath': r'C:\Users\andrew\Documents\localData\vrDatabaseManagement\vrDatabase.accdb',
            'dbName': 'vrDatabase',
            'tableName': 'sessiondb'
        }
    }
    if dbName not in dbdict.keys():
        raise ValueError(f"Did not recognize database={dbName}, valid database names are: {[key for key in dbdict.keys()]}")
    return dbdict[dbName]


class vrDatabase:
    def __init__(self, dbName='vrDatabase'):
        """
        Initialize a new vrDatabase instance.

        This constructor initializes a new instance of the vrDatabase class. It sets the default
        values for the table name, database name, and database path. It is built to work with 
        the Microsoft Access application; however, a few small changes can make it compatible with
        other SQL-based database systems. 

        Parameters
        ----------
        dbName : str, optional
            The name of the database to access. Default is 'vrDatabase'.
            
        Example
        -------
        >>> vrdb = vrDatabase()
        >>> print(vrdb.tableName)
        'sessiondb'
        >>> print(vrdb.dbName)
        'vrDatabase'

        Notes
        -----
        - This constructor uses a supporting function called vrDatabaseMetadata to get database metadata based on the dbName provided.
        - If you are using this on a new system, then you should edit your path, database name, and default table in that function. 
        """
        
        metadata = vrDatabaseMetadata(dbName)
        self.dbPath = metadata['dbPath']
        self.dbName = metadata['dbName']
        self.tableName = metadata['tableName']
        
    def connect(self):
        """
        Connect to the Microsoft Access database defined from the vrDatabaseMetadata function.
        """
        connString = (
            r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
            fr"DBQ={self.dbPath};"
        )
        return pyodbc.connect(connString)
    
    @contextmanager
    def openCursor(self, commitChanges=False):
        """
        Context manager to open a database cursor and manage connections.

        This context manager provides a convenient way to open a cursor to the database,
        perform database operations, and manage connections. It also allows you to
        commit changes if needed.

        Parameters
        ----------
        commitChanges : bool, optional
            Whether to commit changes to the database. Default is False.
        
        Yields
        ------
        pyodbc.Cursor
            A database cursor for executing SQL queries.

        Raises
        ------
        Exception
            If an error occurs while connecting to the database.

        Example
        -------
        Use the context manager to perform database operations:

        >>> with self.openCursor(commitChanges=True) as cursor:
        ...     cursor.execute("SELECT * FROM your_table")

        """
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
    
    # == vrExperiment related methods ==
    def sessionName(self, row):
        mouseName = row['mouseName']
        sessionDate = row['sessionDate'].strftime('%Y-%m-%d')
        sessionID = str(row['sessionID'])
        return mouseName, sessionDate, sessionID

    def vrSession(self, row):
        mouseName, sessionDate, sessionID = self.sessionName(row)
        return vre.vrSession(mouseName, sessionDate, sessionID)
    
    def vrRegistration(self, row, **opts):
        mouseName, sessionDate, sessionID = self.sessionName(row)
        return vre.vrExperimentRegistration(mouseName, sessionDate, sessionID, **opts)
        
    # == retrieve table data ==
    def tableData(self):
        """
        Retrieve data and field names from the specified table.

        This method retrieves the field names and table elements from the table specified
        in the `vrDatabase` instance.

        Returns
        -------
        tuple
            A tuple containing two elements:
            - A list of strings representing the field names of the table.
            - A list of tuples representing the data rows of the table.
        """
        
        with self.openCursor() as cursor:
            fieldNames = [col.column_name for col in cursor.columns(table=self.tableName)]
            cursor.execute(f"SELECT * FROM {self.tableName}")
            tableElements = cursor.fetchall()
            
        return fieldNames, tableElements
    
    def getTable(self, ignoreScratched=True, **kwConditions):
        """
        Retrieve data from table in database and return as dataframe with optional filtering. 
        
        This method retrieves all data from the primary table in the database specified in vrDatabase
        instance. It filters the data to ignore bad sessions (i.e. where sessionQC=False), and can 
        optionally filter based on additional conditions. 
        
        Parameters
        ----------
        ignoreScratched : bool, optional
            Whether to ignore "scratched" sessions. Default is True.
            Scratched sessions are ones where sessionQC=False
        **kwConditions : dict
            Additional filtering conditions as keyword arguments.
            Each condition should match a column name in the table.    
            Note: this is limited in the sense that empty data can't be identified with key:None.
            (using the pd.isnull() is a valid work around, but needs to be coded outside of getTable())
        
        Returns
        -------
        df : pandas dataframe
            A dataframe containing the filtered data from the primary database table.  
            
        Example
        -------
        >>> vrdb = YourDatabaseClass()
        >>> df = vrdb.getTable(ignoreScratched=False, imaging=True)
        """
        
        fieldNames, tableData = self.tableData()
        df = pd.DataFrame.from_records(tableData, columns=fieldNames)
        if ignoreScratched: df = df[df['sessionQC']]
        if kwConditions:
            for key in kwConditions.keys(): 
                assert key in fieldNames, f"{key} is not a column name in {self.tableName}"
            query = " & ".join([f"`{key}`=={val}" for key, val in kwConditions.items()])
            df = df.query(query)
        return df
    
    # == visualization ==
    def showTable(self, table=None):
        show(self.getTable() if table is None else table)
    
    # == helper functions for figuring out what needs work ==
    def needsRegistration(self): 
        df = self.getTable()
        return df[df['vrRegistration']==False]
    
    def updateSuite2pDateTime(self):
        df = self.getTable()
        s2pDone = df[(df['imaging']==True) & (df['suite2p']==True)]
        uids = s2pDone['uSessionID'].tolist()
        s2pCreationDate = []
        for idx, row in s2pDone.iterrows():
            vrs = self.vrSession(row) # create vrSession to point to session folder
            cLatestMod = 0
            for p in vrs.suite2pPath().rglob("*"):
                cLatestMod = max(p.stat().st_mtime, cLatestMod)
            cDateTime = datetime.fromtimestamp(cLatestMod)
            s2pCreationDate.append(cDateTime) # get suite2p path creation date
            
        # return s2pCreationDate
        with self.openCursor(commitChanges=True) as cursor:
            cursor.executemany(self.createUpdateManyStatement('suite2pDate'),zip(s2pCreationDate, uids))
            
    def needsS2P(self):
        df = self.getTable()
        return df[(df['imaging']==True) & (df['suite2p']==False)]
    
    def printRequiresS2P(self, printTargets=True):
        need = self.needsS2P()
        for idx, row in need.iterrows():
            print(f"Database indicates that suite2p has not been run: {self.vrSession(row).sessionPrint()}")
            if printTargets:
                mouseName, sessionDate, sessionID = self.sessionName(row)
                fm.s2pTargets(mouseName, sessionDate, sessionID)
                print("")
      
    # == methods for adding records and updating information to the database ==
    def createUpdateStatement(self, field, uid):
        """to update a single field with a value for a particular session defined by the uSessionID
        
        Example
        -------
        >>> with self.openCursor(commitChanges=True) as cursor:
        >>>     cursor.execute(self.createUpdateManyStatement(<field>, <uid>), <val>)
        """
        return f"UPDATE {self.tableName} set {field} = ? WHERE uSessionID = {uid}"
    
    def createUpdateManyStatement(self, field):
        """to update a single field many times where the value for each uSessionID is provided as a list to cursor.executemany()
        
        Example
        -------
        >>> with self.openCursor(commitChanges=True) as cursor:
        >>>     cursor.executemany(self.createUpdateManyStatement(<field>, [(val,uid),(val,uid),...]))
        """
        return f"UPDATE {self.tableName} set {field} = ? where uSessionID = ?"
    
    def addRecord(self):
        raise ValueError("Not coded yet!")
        #updateStatement = f"insert into T 
        #INSERT INTO Customers (CustomerName, ContactName, Address, City, PostalCode, Country)
        #VALUES ('Cardinal', 'Tom B. Erichsen', 'Skagen 21', 'Stavanger', '4006', 'Norway');
        
    # == operating vrExperiment pipeline ==
    def registerSessions(self, **userOpts):
        # Options for data management:
        # These are a subset of what is available in vre.vrExperimentRegistration 
        # They indicate what preprocessing steps to take depending on what was performed in each experiment
        opts = {}
        opts['vrBehaviorVersion'] = 1 # 1==standard behavioral output (will make conditional loading systems for alternative versions...)
        opts['facecam'] = False # whether or not face video was performed on this session (note: only set to True when DLC has already been run!)
        opts['imaging'] = True # whether or not imaging was performed on this session (note: only set to True when suite2p has already been run!)
        opts['oasis'] = False # whether or not to rerun oasis on calcium signals (note: only used if imaging is run)
        opts['moveRawData'] = False # whether or not to move raw data files to 'rawData'
        opts['redCellProcessing'] = True # whether or not to preprocess redCell features into oneData using the redCellProcessing object (only runs if redcell in self.value['available'])
        opts['clearOne'] = True # clear previous oneData.... yikes, big move dude!
        
        assert userOpts.keys() <= opts.keys(), f"userOpts contains the following invalid keys: {set(userOpts.keys()).difference(opts.keys())}"
        opts.update(userOpts) # Update default opts with user requests
        
        print("In registerSessions, 'vrBehaviorVersion' is an important input that hasn't been coded yet!") 
        
        dfToRegister = self.needsRegistration()
        for idx, row in dfToRegister.iterrows():
            print('')
            opts['imaging']=row['imaging']
            opts['facecam']=row['faceCamera']
            vrExpReg = self.vrRegistration(row, **opts)
            try: 
                print(f"Performing vrExperiment preprocessing for session: {vrExpReg.sessionPrint()}")
                vrExpReg.doPreprocessing()
                print(f"Saving params...")
                vrExpReg.saveParams()
            except Exception as ex:
                print(f"The following exception was raised when trying to preprocess session: {vrExpReg.sessionPrint()}")
                print(f"Exception: {ex}")
            else:
                with self.openCursor(commitChanges=True) as cursor: 
                    # Tell the database that vrRegistration was performed and the time of processing
                    cursor.execute(self.createUpdateStatement('vrRegistration',row['uSessionID']),True)
                    cursor.execute(self.createUpdateStatement('vrRegistrationDate',row['uSessionID']),datetime.now())
            finally: 
                del vrExpReg
                    

class vrDatabaseGrossUpdate(vrDatabase):
    def __init__(self, dbName='vrDatabase'):
        super().__init__(dbName=dbName)
    
    def checkSessionScratch(self, withDatabaseUpdate=False):
        df = self.getTable(ignoreScratched=False)
        good_withJustification = df[(df['sessionQC']==True) & (~pd.isnull(df['scratchJustification']))]
        bad_noJustification = df[(df['sessionQC']==False) & (pd.isnull(df['scratchJustification']))]
        
        goodToBad_UID = good_withJustification['uSessionID'].tolist()
        badToGood_UID = bad_noJustification['uSessionID'].tolist()
        for idx, row in good_withJustification.iterrows():
            print(f"Database said sessionQC=True for {self.vrSession(row).sessionPrint()} but there is a scratchJustification.")
        for idx, row in bad_noJustification.iterrows():
            print(f"Database said sessionQC=False for {self.vrSession(row).sessionPrint()} but there isn't a scratchJustification.")
        
        if withDatabaseUpdate:
            with self.openCursor(commitChanges=True) as cursor: 
                cursor.executemany(self.createUpdateManyStatement('sessionQC'),zip([False]*len(goodToBad_UID),goodToBad_UID))
                cursor.executemany(self.createUpdateManyStatement('sessionQC'),zip([True]*len(badToGood_UID),badToGood_UID))
    
    def checkS2P(self, withDatabaseUpdate=False, returnCheck=False):
        df = self.getTable()
        
        # return dataframe of sessions with imaging where suite2p wasn't done, even though the database thinks it was
        check_s2pDone = df[(df['imaging']==True) & (df['suite2p']==True)]
        checked_notDone = check_s2pDone.apply(lambda row: not(self.vrSession(row).suite2pPath().exists()), axis=1)
        notActuallyDone = check_s2pDone[checked_notDone]
        
        # return dataframe of sessions with imaging where suite2p was done, even though the database thinks it wasn't
        check_s2pNeed = df[(df['imaging']==True) & (df['suite2p']==False)]
        checked_notNeed = check_s2pNeed.apply(lambda row: self.vrSession(row).suite2pPath().exists(), axis=1)
        notActuallyNeed = check_s2pNeed[checked_notNeed]
        
        # Print database errors to workspace
        for idx, row in notActuallyDone.iterrows(): print(f"Database said suite2p has been ran, but it actually hasn't: {self.vrSession(row).sessionPrint()}")
        for idx, row in notActuallyNeed.iterrows(): print(f"Database said suite2p didn't run, but it already did: {self.vrSession(row).sessionPrint()}")
        
        # If withDatabaseUpdate==True, then correct the database
        if withDatabaseUpdate: 
            for idx, row in notActuallyDone.iterrows():
                with self.openCursor(commitChanges=True) as cursor:
                    cursor.execute(self.createUpdateStatement('suite2p',row['uSessionID']),False)
                    
            for idx, row in notActuallyNeed.iterrows():
                with self.openCursor(commitChanges=True) as cursor:
                    cursor.execute(self.createUpdateStatement('suite2p',row['uSessionID']),True)
                    
        # If returnCheck is requested, return True if any records were invalid
        if returnCheck: return checked_notDone.any() or checked_notNeed.any()
    

    
    

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