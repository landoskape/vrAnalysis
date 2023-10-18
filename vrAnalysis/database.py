import traceback
import pyodbc
from IPython.display import Markdown, display
from datetime import datetime
from contextlib import contextmanager
import pandas as pd

from . import session
from . import helpers
from . import fileManagement as fm

def errorPrint(text):
    # supporting function for printing error messages but continuing
    display(Markdown(f'<font color=red>{text}</font>'))

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
        
    def connect(self, hostType='access'):
        """
        Connect to the database defined from the vrDatabaseMetadata function.
        """
        driverString = {
            'access' : r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};" + fr"DBQ={self.dbPath};"
        }
        
        # Make sure connections are possible for this hosttype
        failureMessage = (f"Requested hostType ({hostType}) is not available. The only ones that are coded are: {[k for k in driverString.keys()]}"
                          "For support with writing a driver string for a different host, use the fantastic website: https://www.connectionstrings.com/")
        assert hostType in driverString, failureMessage
        
        # Return a connection to the database
        return pyodbc.connect(driverString[hostType])
    
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
    
    # == vrExperiment related methods ==
    def sessionName(self, row):
        """get session identifiers from record of database"""
        mouseName = row['mouseName']
        sessionDate = row['sessionDate'].strftime('%Y-%m-%d')
        sessionID = str(row['sessionID'])
        return mouseName, sessionDate, sessionID

    def vrSession(self, row):
        """create vrSession object from record in database"""
        mouseName, sessionDate, sessionID = self.sessionName(row)
        return session.vrSession(mouseName, sessionDate, sessionID)
    
    def vrExperiment(self, row):
        """create vrExperiment object from record in database"""
        mouseName, sessionDate, sessionID = self.sessionName(row)
        return session.vrExperiment(mouseName, sessionDate, sessionID)
        
    def vrRegistration(self, row, **opts):
        """create vrRegistration object from record in database"""
        mouseName, sessionDate, sessionID = self.sessionName(row)
        return session.vrRegistration(mouseName, sessionDate, sessionID, **opts)
    
    def miceInSessions(self, iterSession):
        """get list of unique mice names in session iterable"""
        mouseNames = [ses.mouseName for ses in iterSession] 
        return sorted(list(set(mouseNames)))
    
    def printMiceInSessions(self, iterSession):
        """print list of unique mice names in session iterable"""
        print(self.miceInSessions(iterSession))
        
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
            Value can either be a variable (e.g. 0 or 'ATL000'), or a (value, operation) pair.
            The operation defaults to '==', but you can use anything that works as a df query.
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
            for key, val in kwConditions.items(): 
                assert key in fieldNames, f"{key} is not a column name in {self.tableName}"
                if not isinstance(val, tuple):
                    kwConditions[key] = (val, '==') # value, operation pair
            query = " & ".join([f"`{key}`{op}{val!r}" for key, (val, op) in kwConditions.items()])
            df = df.query(query)
        return df
    
    def getRecord(self, mouseName, sessionDate, sessionID):
        """
        Retrieve single record from table in database and return as dataframe. 
        
        This method retrieves a single record(row) from the primary table in the database specified 
        in vrDatabase instance. It identifies which session has the unique combination of mouseName,
        sessionDate, and sessionID, and returns that row. 
        
        Parameters
        ----------
        mouseName : string, required - the name of the mouse, e.g. ATL001
        sessionDate : string, required - the date of the session in yyyy-mm-dd format
        sessionID : int64/string, required - the code for the session, e.g. 701
        
        Returns
        -------
        record : pandas Series
            
        Example
        -------
        >>> vrdb = YourDatabaseClass()
        >>> record = vrdb.getRecord('ATL001','2000-01-01','701')
        """
        
        df = self.getTable()
        record = df[(df['mouseName']==mouseName) 
                    & (df['sessionDate'].apply(lambda sd : sd.strftime('%Y-%m-%d'))==sessionDate)
                    & (df['sessionID']==int(sessionID))]
        if len(record)==0: 
            print(f"No session found under: {mouseName}/{sessionDate}/{sessionID}")
            return None
        if len(record)>1:
            raise ValueError(f"Multiple sessions found under: {mouseName}/{sessionDate}/{sessionID}")
        return record.iloc[0]
    
    def iterSessions(self, ignoreScratched=True, **kwConditions):
        """Creates list of sessions that can be iterated through"""
        df = self.getTable(ignoreScratched=ignoreScratched, **kwConditions)
        return self.createSessionIterable(df)

    def createSessionIterable(self, df):
        ises = []
        for idx, row in df.iterrows():
            ises.append(self.vrExperiment(row))
        return ises
    
    # == visualization ==
    def printSessions(self, ignoreScratched=True, **kwConditions):
        """
        Copy of getTable(), except instead of returning a df, will iterate through the rows and 
        session print each session that meets the given conditions. See getTable()'s documentation
        for info on how to use the optional inputs of this function
        """
        df = self.getTable(ignoreScratched=ignoreScratched, **kwConditions)
        for idx, row in df.iterrows():
            print(self.vrSession(row).sessionPrint())
    
    # == helper functions for figuring out what needs work ==
    def needsRegistration(self, skipErrors=True, as_iterable=True, **kwargs): 
        df = self.getTable(**kwargs)
        if skipErrors: 
            df = df[df['vrRegistrationError']==False]
        df = df[df['vrRegistration']==False]
        if as_iterable: 
            return self.createSessionIterable(df)
        else:
            return df
    
    def updateSuite2pDateTime(self):
        df = self.getTable()
        s2pDone = df[(df['imaging']==True) & (df['suite2p']==True)]
        uids = s2pDone['uSessionID'].tolist()
        s2pCreationDate = []
        for idx, row in s2pDone.iterrows():
            vrs = self.vrSession(row) # create vrSession to point to session folder
            cLatestMod = 0
            for p in vrs.suite2pPath().rglob("*"):
                if not(p.is_dir()):
                    cLatestMod = max(p.stat().st_mtime, cLatestMod)
            cDateTime = datetime.fromtimestamp(cLatestMod)
            s2pCreationDate.append(cDateTime) # get suite2p path creation date
            
        # return s2pCreationDate
        with self.openCursor(commitChanges=True) as cursor:
            cursor.executemany(self.createUpdateManyStatement('suite2pDate'),zip(s2pCreationDate, uids))
            
    def needsS2P(self, needsQC=False):
        df = self.getTable()
        if needsQC:
            return df[(df['imaging']==True) & (df['suite2p']==True) & (df['suite2pQC']==False)]
        else:
            return df[(df['imaging']==True) & (df['suite2p']==False)]
    
    def printRequiresS2P(self, printTargets=True, needsQC=False):
        need = self.needsS2P(needsQC=needsQC)
        for idx, row in need.iterrows():
            if needsQC:
                print(f"Database indicates that suite2p has been run but not QC'd: {self.vrSession(row).sessionPrint()}")
            else:                
                print(f"Database indicates that suite2p has not been run: {self.vrSession(row).sessionPrint()}")
                if printTargets:
                    mouseName, sessionDate, sessionID = self.sessionName(row)
                    fm.s2pTargets(mouseName, sessionDate, sessionID)
                    print("")
    
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
    
    # == for communicating with the database about red cell quality control ==
    def updateRedCellQCDateTime(self):
        relevant_one_files = [
            'mpciROIs.redCellIdx.npy',
            'mpciROIs.redCellManualAssignment.npy',
            'parametersRed*', # wild card because there are multiple possibilities
        ]
            
        df = self.getTable()
        redCellQC_done = df[(df['imaging']==True) & (df['suite2p']==True) & (df['redCellQC']==True)]
        uids = redCellQC_done['uSessionID'].tolist()
        rcEditDate = []
        for idx, row in redCellQC_done.iterrows():
            vrs = self.vrSession(row) # create vrSession to point to session folder
            cLatestMod = 0
            for f in relevant_one_files:
                for file in vrs.onePath().rglob(f):
                    cLatestMod = max(file.stat().st_mtime, cLatestMod)
            cDateTime = datetime.fromtimestamp(cLatestMod)
            rcEditDate.append(cDateTime) # get suite2p path creation date
            
        with self.openCursor(commitChanges=True) as cursor:
            cursor.executemany(self.createUpdateManyStatement('redCellQCDate'),zip(rcEditDate, uids))
            
    def needsRedCellQC(self, **kwConditions):
        df = self.getTable(**kwConditions)
        return df[(df['imaging']==True) & (df['suite2p']==True) & (df['vrRegistration']==True) & (df['redCellQC']==False)]
        
    def printRequiresRedCellQC(self, printTargets=True, needsQC=False, **kwConditions):
        need = self.needsRedCellQC(**kwConditions)
        for idx, row in need.iterrows():
            print(f"Database indicates that redCellQC has not been performed for session: {self.vrSession(row).sessionPrint()}")
    
    def iterSessionNeedRedCellQC(self, **kwConditions):
        """Creates list of sessions that can be iterated through that require red cell quality control"""
        df = self.needsRedCellQC(**kwConditions)
        ises = []
        for idx, row in df.iterrows():
            ises.append(self.vrExperiment(row))
        return ises
    
    def setRedCellQC(self, mouseName, dateString, sessionid, state=True):
        record = self.getRecord(mouseName, dateString, sessionid)
        if record is None: 
            print(f"Could not find session {self.vrSession(record).sessionPrint()} in database.")
            return False
        
        try:
            with self.openCursor(commitChanges=True) as cursor:
                # Tell the database that vrRegistration was performed and the time of processing
                cursor.execute(self.createUpdateStatement('redCellQC',record['uSessionID']),state)
                if state==True:
                    # If saying we are setting red cell qc to true, then add the date
                    cursor.execute(self.createUpdateStatement('redCellQCDate',record['uSessionID']),datetime.now())
                else:
                    # Otherwise remove the date
                    cursor.execute(self.createUpdateStatement('redCellQCDate',record['uSessionID']),'')
            return True
        
        except:
            print(f"Failed to update database for session: {self.vrSession(record).sessionPrint()}")
            return False
    
    # == well, this isn't coded yet :) ==
    def addRecord(self):
        raise ValueError("Not coded yet!")
        #updateStatement = f"insert into T 
        #INSERT INTO Customers (CustomerName, ContactName, Address, City, PostalCode, Country)
        #VALUES ('Cardinal', 'Tom B. Erichsen', 'Skagen 21', 'Stavanger', '4006', 'Norway');
        
    # == operating vrExperiment pipeline ==
    def defaultRegistrationOpts(self, **userOpts):
        # Options for data management:
        # These are a subset of what is available in session.vrRegistration 
        # They indicate what preprocessing steps to take depending on what was performed in each experiment
        opts = {}
        opts['vrBehaviorVersion'] = 1 # 1==standard behavioral output (will make conditional loading systems for alternative versions...)
        opts['facecam'] = False # whether or not face video was performed on this session (note: only set to True when DLC has already been run!)
        opts['imaging'] = True # whether or not imaging was performed on this session (note: only set to True when suite2p has already been run!)
        opts['oasis'] = True # whether or not to rerun oasis on calcium signals (note: only used if imaging is run)
        opts['moveRawData'] = False # whether or not to move raw data files to 'rawData'
        opts['redCellProcessing'] = True # whether or not to preprocess redCell features into oneData using the redCellProcessing object (only runs if redcell in self.value['available'])
        opts['clearOne'] = True # clear previous oneData.... yikes, big move dude!
        
        assert userOpts.keys() <= opts.keys(), f"userOpts contains the following invalid keys: {set(userOpts.keys()).difference(opts.keys())}"
        opts.update(userOpts) # Update default opts with user requests
        return opts
    
    def registerRecord(self, record, **opts):
        opts['imaging'] = bool(record['imaging'])
        opts['facecam'] = bool(record['faceCamera'])
        vrExpReg = self.vrRegistration(record, **opts)
        try: 
            print(f"Performing vrExperiment preprocessing for session: {vrExpReg.sessionPrint()}")
            vrExpReg.doPreprocessing()
            print(f"Saving params...")
            vrExpReg.saveParams()
        except Exception as ex:
            with self.openCursor(commitChanges=True) as cursor: 
                cursor.execute(self.createUpdateStatement('vrRegistrationError',record['uSessionID']), True)
                cursor.execute(self.createUpdateStatement('vrRegistrationException',record['uSessionID']), str(ex))
            print(f"The following exception was raised when trying to preprocess session: {vrExpReg.sessionPrint()}. Clearing all oneData.")
            vrExpReg.clearOneData(certainty=True)
            errorPrint(f"Last traceback: {traceback.extract_tb(ex.__traceback__, limit=-1)}")
            errorPrint(f"Exception: {ex}")
            # If failed, return (False, 0B)
            out = (False, 0)
        else:
            with self.openCursor(commitChanges=True) as cursor: 
                # Tell the database that vrRegistration was performed and the time of processing
                cursor.execute(self.createUpdateStatement('vrRegistration',record['uSessionID']),True)
                cursor.execute(self.createUpdateStatement('vrRegistrationError',record['uSessionID']),False)
                cursor.execute(self.createUpdateStatement('vrRegistrationException',record['uSessionID']),'')
                cursor.execute(self.createUpdateStatement('vrRegistrationDate',record['uSessionID']),datetime.now())
            # If successful, return (True, size of registered oneData)
            out = (True, sum([oneFile.stat().st_size for oneFile in vrExpReg.getSavedOne()])) # accumulate oneData
            print(f"Session {vrExpReg.sessionPrint()} registered with {helpers.readableBytes(out[1])} oneData.")
        finally: 
            del vrExpReg
        return out
        
    def registerSingleSession(self, mouseName, sessionDate, sessionID, **userOpts):
        # get opts for registering session
        opts = self.defaultRegistrationOpts(**userOpts)
        record = self.getRecord(mouseName, sessionDate, sessionID)
        if record is None: 
            print(f"Session {session.vrSession(mouseName, sessionDate, sessionID).sessionPrint()} is not in the database")
            return 
        out = self.registerRecord(record, **opts)
        return out[0]
        
    def registerSessions(self, maxData=30e9, skipErrors=True, **userOpts):
        # get opts for registering session
        opts = self.defaultRegistrationOpts(**userOpts)
        
        print("In registerSessions, 'vrBehaviorVersion' is an important input that hasn't been coded yet!") 
        
        countSessions = 0
        totalOneData = 0.0
        dfToRegister = self.needsRegistration(skipErrors=skipErrors)
        
        for idx, (_, row) in enumerate(dfToRegister.iterrows()):
            if totalOneData > maxData: 
                print(f"\nMax data limit reached. Total processed: {helpers.readableBytes(totalOneData)}. Limit: {helpers.readableBytes(maxData)}")
                return
            print('')
            out = self.registerRecord(row, **opts)
            if out[0]: 
                countSessions += 1 # count successful sessions
                totalOneData += out[1] # accumulated oneData registered
                estimateRemaining = len(dfToRegister) - idx
                print(f"Accumulated oneData registered: {helpers.readableBytes(totalOneData)}. "
                      f"Averaging: {helpers.readableBytes(totalOneData/countSessions)} / session. "
                      f"Estimate remaining: {helpers.readableBytes(totalOneData/countSessions*estimateRemaining)}")
    
    def printRegistrationErrors(self, **kwargs):
        df = self.getTable(**kwargs)
        for idx, row in df[df['vrRegistrationError']==True].iterrows():
            print(f"Session {self.vrRegistration(row).sessionPrint()} had error: {row['vrRegistrationException']}")
            
    # == operating vrExperiment pipeline ==
    def clearOneData(self, **userOpts):
        opts = {}
        opts['clearOne'] = True # clear previous oneData.... yikes, big move dude!
        
        raise ValueError('write in a user prompt to make sure they want to go through with this...')
        
        assert userOpts.keys() <= opts.keys(), f"userOpts contains the following invalid keys: {set(userOpts.keys()).difference(opts.keys())}"
        opts.update(userOpts) # Update default opts with user requests
        
        print("In registerSessions, 'vrBehaviorVersion' is an important input that hasn't been coded yet!") 
        
        dfToRegister = self.needsRegistration()
        for idx, row in dfToRegister.iterrows():
            print('')
            vrExpReg = self.vrRegistration(row, **opts)
            try: 
                print(f"Removing oneData for session: {vrExpReg.sessionPrint()}")
                vrExpReg.clearOneData()
                vrExpReg.preprocessing = []
                vrExpReg.saveParams()
            except Exception as ex:
                print(f"The following exception was raised when trying to preprocess session: {vrExpReg.sessionPrint()}")
                print(f"Exception: {ex}")
            else:
                with self.openCursor(commitChanges=True) as cursor: 
                    # Tell the database that vrRegistration was performed and the time of processing
                    cursor.execute(self.createUpdateStatement('vrRegistration',row['uSessionID']),None)
                    cursor.execute(self.createUpdateStatement('vrRegistrationDate',row['uSessionID']),None)
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