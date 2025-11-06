import traceback
import pyodbc
from datetime import datetime, date
from contextlib import contextmanager
from pathlib import Path
import pandas as pd
from copy import copy
from subprocess import run
from typing import Union

# Back-compatibility imports
from _old_vrAnalysis import session as original_session
from _old_vrAnalysis import registration
from _old_vrAnalysis import fileManagement as fm

# vrAnalysis imports
from .helpers import readable_bytes, error_print, get_confirmation
from .sessions import B2Session


def get_database_metadata(db_name: str) -> dict:
    """
    Retrieve metadata for a specified database.

    This function retrieves metadata for a specified database from the `dbdict` dictionary.
    The `dbdict` dictionary contains the database paths, names, and primary table name.

    Parameters
    ----------
    db_name : str
        The name of the database for which to retrieve metadata.

    Returns
    -------
    dict
        A dictionary containing metadata for the specified database.
        It requires the following keys:
            'db_path': path to the database file
            'db_name': name of the database file
            'db_ext': extension of the database file
            'table_name': name of the table to use
            'uid': name of the field defining a unique ID for each row in the table
            'backup_path': path to the database backup (None if there isn't one)
            'unique_fields': list of names of fields for which there should only be one
                            database row per combination of the values in unique_fields
                            note: assumes string, but make it a tuple for different types
            'default_conditions': dictionary containing key-value pairs of any default
                                 conditions to filter by when retrieving table data

    Raises
    ------
    ValueError
        If the provided `dbName` is not recognized as a valid database name.

    Example
    -------
    >>> metadata = vrDatabaseMetadata('vrSessions')
    >>> print(metadata['db_path'])
    'C:\\Users\\andrew\\Documents\\localData\\vrDatabaseManagement'
    >>> print(metadata['db_name'])
    'vrDatabase'

    Notes
    -----
    - The `dbdict` dictionary contains metadata for recognized databases.
    - Edit this function to specify the path, database, and primary table on your computer.
    """

    dbdict = {
        "vrSessions": {
            "db_path": r"C:\Users\andrew\Documents\localData\vrDatabaseManagement",
            "db_name": "vrDatabase",
            "db_ext": ".accdb",
            "table_name": "sessiondb",
            "uid": "uSessionID",
            "backup_path": r"D:\localData\vrDatabaseManagement",
            "unique_fields": [("mouseName", str), ("sessionDate", datetime), ("sessionID", int)],
            "default_conditions": {
                "sessionQC": True,
            },
            "constructor": SessionDatabase,
        },
        "vrMice": {
            "db_path": r"C:\Users\andrew\Documents\localData\vrDatabaseManagement",
            "db_name": "vrDatabase",
            "db_ext": ".accdb",
            "table_name": "mousedb",
            "uid": "uMouseID",
            "backup_path": r"D:\localData\vrDatabaseManagement",
            "unique_fields": [("mouseName", str)],
            "default_conditions": {},
            "constructor": BaseDatabase,
        },
    }
    if db_name not in dbdict.keys():
        raise ValueError(f"Did not recognize database={db_name}, valid database names are: {[key for key in dbdict.keys()]}")
    return dbdict[db_name]


# a dictionary of host types determining what driver string to use for database connections
host_types = {
    ".accdb": "access",
    ".mdb": "access",
}


def get_database(db_name: str) -> Union["BaseDatabase", "SessionDatabase"]:
    """
    Method for retrieving an appropriate database object.
    """
    metadata = get_database_metadata(db_name)
    if "constructor" in metadata:
        if issubclass(metadata["constructor"], BaseDatabase):
            constructor = metadata["constructor"]  # get class constructor method for this database
        else:
            raise ValueError(f"{metadata['constructor']} must be a subclass of the `BaseDatabase` class!")
    else:
        constructor = BaseDatabase
    return constructor(db_name)


class BaseDatabase:
    def __init__(self, db_name: str):
        """
        Initialize a new database instance.

        This constructor initializes a new instance of the BaseDatabase class. It sets the default
        values for the table name, database name, and database path. It is built to work with
        the Microsoft Access application; however, a few small changes can make it compatible with
        other SQL-based database systems.

        Parameters
        ----------
        db_name : str, required
            The name of the database to access.

        Example
        -------
        >>> db = BaseDatabase('vrSessions')
        >>> print(vrdb.table_name)
        'sessiondb'
        >>> print(vrdb.db_name)
        'vrDatabase'

        Notes
        -----
        - This constructor uses a supporting function called vrDatabaseMetadata to get database metadata based on the dbName provided.
        - If you are using this on a new system, then you should edit your path, database name, and default table in that function.
        """

        metadata = get_database_metadata(db_name)
        self.db_path = metadata["db_path"]
        self.db_name = metadata["db_name"]
        self.db_ext = metadata["db_ext"]
        self.table_name = metadata["table_name"]
        self.uid = metadata["uid"]
        self.backup_path = metadata["backup_path"]
        self.host_type = host_types[self.db_ext]
        self.unique_fields = self.process_unique_fields(metadata["unique_fields"])
        self.default_conditions = metadata["default_conditions"]

    def process_unique_fields(self, fields):
        ufields = []
        for f in fields:
            if isinstance(f, tuple) and len(f) == 2 and type(f[1]) == type and isinstance(f[0], str):
                ufields.append(f)
            elif isinstance(f, str):
                ufields.append((f, str))
            else:
                raise ValueError(f"unique field {f} must be a string or a string-type tuple")
        return ufields

    def get_dbfile(self):
        return Path(self.db_path) / (self.db_name + self.db_ext)

    def save_backup(self, return_out: bool = False) -> str | None:
        """Save a backup of the database to the backup path specified in metadata.

        Parameters
        ----------
        return_out : bool, optional
            If True, return the output of the robocopy command.

        Returns
        -------
        str | None
            Output from the robocopy command (if return_out is True).
        """
        source_path = self.db_path
        target_path = self.backup_path
        source_file = self.db_name + self.db_ext
        robocopy_arguments = f"robocopy {source_path} {target_path} {source_file}"
        outs = run(robocopy_arguments, capture_output=True, text=True)
        if return_out:
            return outs

    def connect(self):
        """
        Connect to the database defined from the vrDatabaseMetadata function.

        This is built to work with different database managers, but unless you are using
        Microsoft Access, you'll need to do a little work. First, figure out how to write
        a driverString for the pyodbc connection. You can use this website for help:

        https://www.connectionstrings.com/

        After making a new driver string, add it to the driver string dictionary and choose
        a unique key. Once you've chosen your key, add the key to the dictionary called
        "host_types" at the top of this file as the value associated with the file extension
        for your database.

        Note: I am assuming that the file extension will fully specify the host type and driver
        string. If this isn't the case and you're not sure what to do, contact me so we can make
        a better system.
        """
        driver_string = {"access": r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};" + rf"DBQ={self.get_dbfile()};"}

        # Make sure connections are possible for this hosttype
        failure_message = (
            f"Requested host_type ({self.host_type}) is not available. The only ones that are coded are: {[k for k in driver_string.keys()]}\n\n"
            f"For support with writing a driver string for a different host, use the fantastic website: https://www.connectionstrings.com/"
        )
        assert self.host_type in driver_string, failure_message

        # Return a connection to the database
        return pyodbc.connect(driver_string[self.host_type])

    @contextmanager
    def open_cursor(self, commit_changes: bool = False):
        """
        Context manager to open a database cursor and manage connections.

        This context manager provides a convenient way to open a cursor to the database,
        perform database operations, and manage connections. It also allows you to
        commit changes if needed.

        Parameters
        ----------
        commit_changes : bool, optional
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

        >>> with self.open_cursor(commit_changes=True) as cursor:
        ...     cursor.execute("SELECT * FROM your_table")

        """
        try:
            # Attempt to open a cursor to the database
            conn = self.connect()
            cursor = conn.cursor()
            yield cursor
        except Exception as ex:
            print(f"An exception occurred while trying to connect to {self.dbName}!")
            print(ex)
            raise ex
        else:
            # if no exception was raised, commit changes
            if commit_changes:
                conn.commit()
        finally:
            # Always close the cursor and connection
            cursor.close()
            conn.close()

    # == display meta data for database ==
    def show_metadata(self):
        """convience method for showing the metadata associated with the open database"""
        print(f"{self.host_type} database located at {self.db_path}")
        print(f"Database name: {self.db_name}{self.db_ext}, table name: {self.table_name}, with uid: {self.uid}")
        if self.backup_path is not None:
            print(f"Backup path located at: {self.backup_path}")
        else:
            print(f"No backup path specified...")
        if self.default_conditions:
            print(f"Default database filters:")
            for key, val in self.default_conditions.items():
                print("  ", self.construct_filter_string(key, self.process_filter_value(val)))
        else:
            print(f"No default filters.")

    # == retrieve table data ==
    def table_data(self):
        """
        Retrieve data and field names from the specified table.

        This method retrieves the field names and table elements from the table specified
        in the `BaseDatabase` instance.

        Returns
        -------
        tuple
            A tuple containing two elements:
            - A list of strings representing the field names of the table.
            - A list of tuples representing the data rows of the table.
        """
        with self.open_cursor() as cursor:
            fieldNames = [col.column_name for col in cursor.columns(table=self.table_name)]
            cursor.execute(f"SELECT * FROM {self.table_name}")
            tableElements = cursor.fetchall()

        return fieldNames, tableElements

    def get_table(self, use_default: bool = True, **kwConditions):
        """
        Retrieve data from table in database and return as dataframe with optional filtering.

        This method retrieves all data from the primary table in the database specified in
        BaseDatabase instance. It automatically filters the data using the defaultConditions
        defined in the dbMetadata method. kwConditions overwrite defaultConditions if there is
        a conflict.

        Parameters
        ----------
        use_default : bool, default=True
            Use default conditions if true, if False ignore them
        **kwConditions : dict
            Additional filtering conditions as keyword arguments.
            Each condition should match a column name in the table.
            Value can either be a variable (e.g. 0 or 'ATL000'), or a (value, operation) pair.
            The operation defaults to '==', but you can use anything that works as a df query.
            Note: this is limited in the sense that empty data can't be identified with key:None.
            (using the pd.isnull() is a valid work around, but needs to be coded outside of get_table())

        Returns
        -------
        df : pandas dataframe
            A dataframe containing the filtered data from the primary database table.

        Example
        -------
        >>> vrdb = YourDatabaseClass()
        >>> df = vrdb.get_table(imaging=True)
        """

        fieldNames, table_data = self.table_data()
        df = pd.DataFrame.from_records(table_data, columns=fieldNames)
        conditions = copy(self.defaultConditions) if use_default else {}
        conditions.update(kwConditions)
        if conditions:
            for key, val in conditions.items():
                assert key in fieldNames, f"{key} is not a column name in {self.table_name}"
                conditions[key] = self.process_filter_value(val)  # make sure it's a value/operation pair
            query = " & ".join([self.construct_filter_string(key, val_op_tuple) for key, val_op_tuple in conditions.items()])
            df = df.query(query)
        return df

    def process_filter_value(self, val):
        """
        Make sure filter value has an operation associated with it.

        Filters are passed to the database as {key}{operation}{value}
        Therefore, each value in the filter conditions needs to be a tuple in which
        the first element is the value and the second element is the operation.
        """
        if not isinstance(val, tuple):
            val = (val, "==")
        return val

    def construct_filter_string(self, key, val_op_tuple):
        """constructs a string to be used as a dataframe query"""
        val, op = val_op_tuple
        return f"`{key}`{op}{val!r}"

    # == methods for adding records and updating information to the database ==
    def createUpdateStatement(self, field, uid):
        """to update a single field with a value for a particular session defined by the uid

        Example
        -------
        >>> with self.open_cursor(commit_changes=True) as cursor:
        >>>     cursor.execute(self.createUpdateManyStatement(<field>, <uid>), <val>)
        """
        return f"UPDATE {self.table_name} set {field} = ? WHERE {self.uid} = {uid}"

    def createUpdateManyStatement(self, field):
        """to update a single field many times where the value for each uid is provided as a list to cursor.executemany()

        Example
        -------
        >>> with self.open_cursor(commit_changes=True) as cursor:
        >>>     cursor.executemany(self.createUpdateManyStatement(<field>, [(val,uid),(val,uid),...]))
        """
        return f"UPDATE {self.table_name} set {field} = ? where {self.uid} = ?"

    def updateDatabaseField(self, field, val, **kwConditions):
        """
        Method for updating a database field in every record where conditions are true

        The ignoreScratched and kwConditions arguments are fed into the self.get_table() method
        Then, every record that is returned gets <field> updated to <val>
        """
        assert field in self.table_data()[0], f"Requested field ({field}) is not in table. Use 'self.table_data()[0]' to see available fields."
        df = self.get_table(**kwConditions)
        updateStatement = self.createUpdateManyStatement(field)
        uids = df[self.uid].tolist()  # uids of all sessions requested
        val_as_list = [val] * len(uids)  #
        print(f"Setting {field}={val} for all requested records...")
        with self.open_cursor(commit_changes=True) as cursor:
            cursor.executemany(updateStatement, zip(val_as_list, uids))

    # == method for adding a record to the database ==
    def addRecord(self, insert_statement, columns, values):
        """
        Attempt to add a single record to the database

        First checks if the values associated with uniqueFields match an existing record, and
        prevent the record from being added if so.

        Otherwise, adds the record to the database.
        """
        d = dict(zip(columns, values))
        unique_values = [d[uf[0]] for uf in self.unique_fields]  # get values associated with unique fields
        for ii, uv in enumerate(unique_values):
            if isinstance(uv, date) or isinstance(uv, datetime):
                # this is required for communicating with Access
                unique_values[ii] = uv.strftime("%Y-%m-%d")
        unique_combo = ", ".join([f"{uf[0]}={uv}" for uf, uv in zip(self.unique_fields, unique_values)])
        if self.getRecord(*unique_values, verbose=False) is not None:
            print(f"Record already exists for {unique_combo}")
            return f"Record already exists for {unique_combo}"
        with self.open_cursor(commit_changes=True) as cursor:
            cursor.execute(insert_statement, values)
            print(f"Successfully added new record for {unique_combo}")
        return "Successfully added new record"

    def getRecord(self, *unique_values, verbose=True):
        """
        Retrieve single record from table in database and return as dataframe.

        This method retrieves a single record(row) from the table in the database. The metadata for
        each database defines a set of fields that comprise a unique set (each combination of values
        for the unique fields is only represented once in the database).

        Parameters
        ----------
        *unique_values: variable length list of values associated with the unique fields
            - must be the same length as self.uniqueFields
            - the second value of the uniqueField tuple (string by default) determines how
              to query the unique value

        Returns
        -------
        record : pandas Series

        Example
        -------
        >>> vrdb = YourDatabaseClass()
        >>> record = vrdb.getRecord(*uniqueConditions)
        """

        # Check if correct values are provided (or if a session object is provided)
        if len(unique_values) == 1 and isinstance(unique_values[0], original_session.vrSession):
            unique_values = [unique_values[0].mouseName, unique_values[0].dateString, unique_values[0].sessionid]

        elif len(unique_values) != len(self.unique_fields):
            expected_list = ", ".join([uf[0] for uf in self.unique_fields])
            raise ValueError(f"{len(unique_values)} values provided but *getRecord* is expecting values for: {expected_list}")

        # Get table and compare
        df = self.get_table()
        for uf, uv in zip(self.unique_fields, unique_values):
            if uf[1] == str:
                df = df[df[uf[0]] == uv]
            elif uf[1] == datetime:
                df = df[df[uf[0]].apply(lambda sd: sd.strftime("%Y-%m-%d")) == uv]
            elif uf[1] == int:
                df = df[df[uf[0]] == int(uv)]
            else:
                raise ValueError(f"uniqueField type ({uf[1]}) not recognized, add the appropriate query to this method!")

        if len(df) == 0:
            if verbose:
                unique_combo = ", ".join([f"{uf[0]}={uv}" for uf, uv in zip(self.uniqueFields, unique_values)])
                print(f"No session found under: {unique_combo}")
            return None
        if len(df) > 1:
            unique_combo = ", ".join([f"{uf[0]}={uv}" for uf, uv in zip(self.uniqueFields, unique_values)])
            raise ValueError(f"Multiple sessions found under: {unique_combo}")
        return df.iloc[0]


# ======== child database class definition for sessions ==========
class SessionDatabase(BaseDatabase):
    """child database for handling vrSessions"""

    def iter_sessions(self, session_params: dict = {}, **kw_conditions) -> list[B2Session]:
        """Iterate over sessions matching conditions.

        Parameters
        ----------
        session_params : dict, default={}
            Additional parameters to pass to the session constructor.
        **kw_conditions : dict
            Additional conditions to filter the sessions by.

        Returns
        -------
        sessions : list[B2Session]
            List of sessions matching the conditions.
        """
        df = self.get_table(**kw_conditions)
        sessions = []
        for _, row in df.iterrows():
            sessions.append(B2Session.create(row["mouseName"], row["sessionDate"], str(row["sessionID"]), params=session_params))
        return sessions

    # == EVERYTHING BELOW HERE IS THE SAME AS THE ORIGINAL DATABASE CLASS ==
    # == It should be refactored to use the new vrAnalysis classes eventually, but I'm leaving it here for now ==
    # == It should work as is as long as vrAnalysis stays on the path! ==

    # == vrExperiment related methods ==
    def sessionName(self, row):
        """get session identifiers from record of database"""
        mouseName = row["mouseName"]
        sessionDate = row["sessionDate"].strftime("%Y-%m-%d")
        sessionID = str(row["sessionID"])
        return mouseName, sessionDate, sessionID

    def vrSession(self, row):
        """create vrSession object from record in database"""
        mouseName, sessionDate, sessionID = self.sessionName(row)
        return original_session.vrSession(mouseName, sessionDate, sessionID)

    def vrExperiment(self, row):
        """create vrExperiment object from record in database"""
        mouseName, sessionDate, sessionID = self.sessionName(row)
        return original_session.vrExperiment(mouseName, sessionDate, sessionID)

    def vrRegistration(self, row, **opts):
        """create vrRegistration object from record in database"""
        mouseName, sessionDate, sessionID = self.sessionName(row)
        return registration.vrRegistration(mouseName, sessionDate, sessionID, **opts)

    def miceInSessions(self, iterSession):
        """get list of unique mice names in session iterable"""
        mouseNames = [ses.mouseName for ses in iterSession]
        return sorted(list(set(mouseNames)))

    def printMiceInSessions(self, iterSession):
        """print list of unique mice names in session iterable"""
        print(self.miceInSessions(iterSession))

    def iterSessions(self, **kwConditions) -> list[original_session.vrSession]:
        """Creates list of sessions that can be iterated through"""
        df = self.get_table(**kwConditions)
        return self.createSessionIterable(df)

    def createSessionIterable(self, df, session_constructor=None):
        if session_constructor is None:
            session_constructor = self.vrExperiment
        ises = []
        for idx, row in df.iterrows():
            ises.append(session_constructor(row))
        return ises

    # == visualization ==
    def printSessions(self, **kwConditions):
        """
        Copy of get_table(), except instead of returning a df, will iterate through the rows and
        session print each session that meets the given conditions. See get_table()'s documentation
        for info on how to use the optional inputs of this function
        """
        df = self.get_table(**kwConditions)
        for _, row in df.iterrows():
            print(self.vrSession(row).sessionPrint())

    # == helper functions for figuring out what needs work ==
    def needsRegistration(self, skipErrors=True, as_iterable=False, **kwargs):
        df = self.get_table(**kwargs)
        if skipErrors:
            df = df[df["vrRegistrationError"] == False]
        df = df[df["vrRegistration"] == False]
        if as_iterable:
            return self.createSessionIterable(df, session_constructor=self.vrRegistration)
        else:
            return df

    def updateSuite2pDateTime(self):
        df = self.get_table()
        s2pDone = df[(df["imaging"] == True) & (df["suite2p"] == True)]
        uids = s2pDone[self.uid].tolist()
        s2pCreationDate = []
        for idx, row in s2pDone.iterrows():
            vrs = self.vrSession(row)  # create vrSession to point to session folder
            cLatestMod = 0
            for p in vrs.suite2pPath().rglob("*"):
                if not (p.is_dir()):
                    cLatestMod = max(p.stat().st_mtime, cLatestMod)
            cDateTime = datetime.fromtimestamp(cLatestMod)
            s2pCreationDate.append(cDateTime)  # get suite2p path creation date

        # return s2pCreationDate
        with self.open_cursor(commit_changes=True) as cursor:
            cursor.executemany(self.createUpdateManyStatement("suite2pDate"), zip(s2pCreationDate, uids))

    def needsS2P(self, needsQC=False):
        df = self.get_table()
        if needsQC:
            return df[(df["imaging"] == True) & (df["suite2p"] == True) & (df["suite2pQC"] == False)]
        else:
            return df[(df["imaging"] == True) & (df["suite2p"] == False)]

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
        df = self.get_table()

        # return dataframe of sessions with imaging where suite2p wasn't done, even though the database thinks it was
        check_s2pDone = df[(df["imaging"] == True) & (df["suite2p"] == True)]
        checked_notDone = check_s2pDone.apply(lambda row: not (self.vrSession(row).suite2pPath().exists()), axis=1)
        notActuallyDone = check_s2pDone[checked_notDone]

        # return dataframe of sessions with imaging where suite2p was done, even though the database thinks it wasn't
        check_s2pNeed = df[(df["imaging"] == True) & (df["suite2p"] == False)]
        checked_notNeed = check_s2pNeed.apply(lambda row: self.vrSession(row).suite2pPath().exists(), axis=1)
        notActuallyNeed = check_s2pNeed[checked_notNeed]

        # Print database errors to workspace
        for idx, row in notActuallyDone.iterrows():
            print(f"Database said suite2p has been ran, but it actually hasn't: {self.vrSession(row).sessionPrint()}")
        for idx, row in notActuallyNeed.iterrows():
            print(f"Database said suite2p didn't run, but it already did: {self.vrSession(row).sessionPrint()}")

        # If withDatabaseUpdate==True, then correct the database
        if withDatabaseUpdate:
            for idx, row in notActuallyDone.iterrows():
                with self.open_cursor(commit_changes=True) as cursor:
                    cursor.execute(self.createUpdateStatement("suite2p", row[self.uid]), False)

            for idx, row in notActuallyNeed.iterrows():
                with self.open_cursor(commit_changes=True) as cursor:
                    cursor.execute(self.createUpdateStatement("suite2p", row[self.uid]), True)

        # If returnCheck is requested, return True if any records were invalid
        if returnCheck:
            return checked_notDone.any() or checked_notNeed.any()

    # == for communicating with the database about red cell quality control ==
    def updateRedCellQCDateTime(self):
        relevant_one_files = [
            "mpciROIs.redCellIdx.npy",
            "mpciROIs.redCellManualAssignment.npy",
            "parametersRed*",  # wild card because there are multiple possibilities
        ]

        df = self.get_table()
        redCellQC_done = df[(df["imaging"] == True) & (df["suite2p"] == True) & (df["redCellQC"] == True)]
        uids = redCellQC_done[self.uid].tolist()
        rcEditDate = []
        for idx, row in redCellQC_done.iterrows():
            vrs = self.vrSession(row)  # create vrSession to point to session folder
            cLatestMod = 0
            for f in relevant_one_files:
                for file in vrs.onePath().rglob(f):
                    cLatestMod = max(file.stat().st_mtime, cLatestMod)
            cDateTime = datetime.fromtimestamp(cLatestMod)
            rcEditDate.append(cDateTime)  # get suite2p path creation date

        with self.open_cursor(commit_changes=True) as cursor:
            cursor.executemany(self.createUpdateManyStatement("redCellQCDate"), zip(rcEditDate, uids))

    def needsRedCellQC(self, **kwConditions):
        df = self.get_table(**kwConditions)
        return df[(df["imaging"] == True) & (df["suite2p"] == True) & (df["vrRegistration"] == True) & (df["redCellQC"] == False)]

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
            with self.open_cursor(commit_changes=True) as cursor:
                # Tell the database that vrRegistration was performed and the time of processing
                cursor.execute(self.createUpdateStatement("redCellQC", record[self.uid]), state)
                if state == True:
                    # If saying we are setting red cell qc to true, then add the date
                    cursor.execute(
                        self.createUpdateStatement("redCellQCDate", record[self.uid]),
                        datetime.now(),
                    )
                else:
                    # Otherwise remove the date
                    cursor.execute(self.createUpdateStatement("redCellQCDate", record[self.uid]), "")
            return True

        except:
            print(f"Failed to update database for session: {self.vrSession(record).sessionPrint()}")
            return False

    # == operating vrExperiment pipeline ==
    def defaultRegistrationOpts(self, **userOpts):
        # Options for data management:
        # These are a subset of what is available in session.vrRegistration
        # They indicate what preprocessing steps to take depending on what was performed in each experiment
        opts = {}
        opts["vrBehaviorVersion"] = 1  # 1==standard behavioral output (will make conditional loading systems for alternative versions...)
        opts["facecam"] = False  # whether or not face video was performed on this session (note: only set to True when DLC has already been run!)
        opts["imaging"] = True  # whether or not imaging was performed on this session (note: only set to True when suite2p has already been run!)
        opts["oasis"] = True  # whether or not to rerun oasis on calcium signals (note: only used if imaging is run)
        opts["redCellProcessing"] = (
            True  # whether or not to preprocess redCell features into oneData using the redCellProcessing object (only runs if redcell in self.value['available'])
        )
        opts["clearOne"] = True  # clear previous oneData.... yikes, big move dude!

        assert userOpts.keys() <= opts.keys(), f"userOpts contains the following invalid keys: {set(userOpts.keys()).difference(opts.keys())}"
        opts.update(userOpts)  # Update default opts with user requests
        return opts

    def registerRecord(self, record, raise_exception=False, **opts):
        opts["imaging"] = bool(record["imaging"])
        opts["facecam"] = bool(record["faceCamera"])
        opts["vrBehaviorVersion"] = record["vrBehaviorVersion"]
        vrExpReg = self.vrRegistration(record, **opts)
        try:
            print(f"Performing vrExperiment preprocessing for session: {vrExpReg.sessionPrint()}")
            vrExpReg.doPreprocessing()
            print(f"Saving params...")
            vrExpReg.saveParams()
        except Exception as ex:
            with self.open_cursor(commit_changes=True) as cursor:
                cursor.execute(self.createUpdateStatement("vrRegistrationError", record[self.uid]), True)
                cursor.execute(
                    self.createUpdateStatement("vrRegistrationException", record[self.uid]),
                    str(ex),
                )
            if raise_exception:
                raise ex
            print(f"The following exception was raised when trying to preprocess session: {vrExpReg.sessionPrint()}. Clearing all oneData.")
            vrExpReg.clearOneData(certainty=True)
            error_print(f"Last traceback: {traceback.extract_tb(ex.__traceback__, limit=-1)}")
            error_print(f"Exception: {ex}")
            # If failed, return (False, 0B)
            out = (False, 0)
        else:
            with self.open_cursor(commit_changes=True) as cursor:
                # Tell the database that vrRegistration was performed and the time of processing
                cursor.execute(self.createUpdateStatement("vrRegistration", record[self.uid]), True)
                cursor.execute(self.createUpdateStatement("vrRegistrationError", record[self.uid]), False)
                cursor.execute(self.createUpdateStatement("vrRegistrationException", record[self.uid]), "")
                cursor.execute(
                    self.createUpdateStatement("vrRegistrationDate", record[self.uid]),
                    datetime.now(),
                )
            # If successful, return (True, size of registered oneData)
            out = (
                True,
                sum([oneFile.stat().st_size for oneFile in vrExpReg.getSavedOne()]),
            )  # accumulate oneData
            print(f"Session {vrExpReg.sessionPrint()} registered with {readable_bytes(out[1])} oneData.")
        finally:
            del vrExpReg
        return out

    def registerSingleSession(self, mouseName, sessionDate, sessionID, raise_exception=False, **userOpts):
        # get opts for registering session
        opts = self.defaultRegistrationOpts(**userOpts)
        record = self.getRecord(mouseName, sessionDate, sessionID)
        if record is None:
            print(f"Session {original_session.vrSession(mouseName, sessionDate, sessionID).sessionPrint()} is not in the database")
            return
        out = self.registerRecord(record, raise_exception=raise_exception, **opts)
        return out[0]

    def registerSessions(self, maxData=30e9, skipErrors=True, raise_exception=False, **userOpts):
        # get opts for registering session
        opts = self.defaultRegistrationOpts(**userOpts)

        countSessions = 0
        totalOneData = 0.0
        dfToRegister = self.needsRegistration(skipErrors=skipErrors)

        for idx, (_, row) in enumerate(dfToRegister.iterrows()):
            if totalOneData > maxData:
                print(f"\nMax data limit reached. Total processed: {readable_bytes(totalOneData)}. Limit: {readable_bytes(maxData)}")
                return
            print("")
            out = self.registerRecord(row, raise_exception=raise_exception, **opts)
            if out[0]:
                countSessions += 1  # count successful sessions
                totalOneData += out[1]  # accumulated oneData registered
                estimateRemaining = len(dfToRegister) - idx - 1
                print(
                    f"Accumulated oneData registered: {readable_bytes(totalOneData)}. "
                    f"Averaging: {readable_bytes(totalOneData/countSessions)} / original_session. "
                    f"Estimate remaining: {readable_bytes(totalOneData/countSessions*estimateRemaining)}"
                )

    def printRegistrationErrors(self, **kwargs):
        df = self.get_table(**kwargs)
        for idx, row in df[df["vrRegistrationError"] == True].iterrows():
            print(f"Session {self.vrRegistration(row).sessionPrint()} had error: {row['vrRegistrationException']}")

    # == operating vrExperiment pipeline ==
    def clearOneData(self, **userOpts):
        opts = {}
        opts["clearOne"] = True  # clear previous oneData.... yikes, big move dude!

        confirmation = get_confirmation("Are you sure you want to clear oneData for all sessions?")
        if not confirmation:
            return

        assert userOpts.keys() <= opts.keys(), f"userOpts contains the following invalid keys: {set(userOpts.keys()).difference(opts.keys())}"
        opts.update(userOpts)  # Update default opts with user requests

        print("In registerSessions, 'vrBehaviorVersion' is an important input that hasn't been coded yet!")

        dfToRegister = self.needsRegistration()
        for idx, row in dfToRegister.iterrows():
            print("")
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
                with self.open_cursor(commit_changes=True) as cursor:
                    # Tell the database that vrRegistration was performed and the time of processing
                    cursor.execute(self.createUpdateStatement("vrRegistration", row[self.uid]), None)
                    cursor.execute(self.createUpdateStatement("vrRegistrationDate", row[self.uid]), None)
            finally:
                del vrExpReg


class vrDatabaseGrossUpdate(SessionDatabase):
    def __init__(self, dbName="vrDatabase"):
        super().__init__(dbName=dbName)

    def checkSessionScratch(self, withDatabaseUpdate=False):
        df = self.get_table(ignoreScratched=False)
        good_withJustification = df[(df["sessionQC"] == True) & (~pd.isnull(df["scratchJustification"]))]
        bad_noJustification = df[(df["sessionQC"] == False) & (pd.isnull(df["scratchJustification"]))]

        goodToBad_UID = good_withJustification[self.uid].tolist()
        badToGood_UID = bad_noJustification[self.uid].tolist()
        for idx, row in good_withJustification.iterrows():
            print(f"Database said sessionQC=True for {self.vrSession(row).sessionPrint()} but there is a scratchJustification.")
        for idx, row in bad_noJustification.iterrows():
            print(f"Database said sessionQC=False for {self.vrSession(row).sessionPrint()} but there isn't a scratchJustification.")

        if withDatabaseUpdate:
            with self.open_cursor(commit_changes=True) as cursor:
                cursor.executemany(
                    self.createUpdateManyStatement("sessionQC"),
                    zip([False] * len(goodToBad_UID), goodToBad_UID),
                )
                cursor.executemany(
                    self.createUpdateManyStatement("sessionQC"),
                    zip([True] * len(badToGood_UID), badToGood_UID),
                )


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
