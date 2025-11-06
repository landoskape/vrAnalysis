"""
Database management module for VR analysis sessions.

This module provides classes and functions for interacting with Microsoft Access
databases used to track VR session data. It includes base database functionality
and specialized session database management with support for registration workflows,
suite2p processing tracking, and quality control operations.
"""

import traceback
import pyodbc
from datetime import datetime, date
from contextlib import contextmanager
from pathlib import Path
import pandas as pd
from copy import copy
from subprocess import run, CompletedProcess
from typing import Union, List, Tuple, Optional, Any, Dict, Generator

from .files import s2p_targets
from .helpers import readable_bytes, error_print
from .sessions import B2Session
from .sessions.b2session import B2RegistrationOpts
from .registration import B2Registration


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
        If the provided `db_name` is not recognized as a valid database name.

    Example
    -------
    >>> metadata = get_database_metadata('vrSessions')
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


def get_database(db_name: str) -> Union["BaseDatabase", "SessionDatabase"]:
    """
    Retrieve an appropriate database object instance.

    This function retrieves metadata for the specified database and instantiates
    the appropriate database class (BaseDatabase or a subclass like SessionDatabase).

    Parameters
    ----------
    db_name : str
        The name of the database to retrieve.

    Returns
    -------
    BaseDatabase or SessionDatabase
        An instance of the appropriate database class as specified in the metadata.

    Raises
    ------
    ValueError
        If the constructor specified in metadata is not a subclass of BaseDatabase.

    See Also
    --------
    get_database_metadata : Retrieve metadata for a database.
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


# a dictionary of host types determining what driver string to use for database connections
host_types = {
    ".accdb": "access",
    ".mdb": "access",
}


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
        - This constructor uses a supporting function called get_database_metadata to get database metadata based on the db_name provided.
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

    def process_unique_fields(self, fields: List[Union[str, Tuple[str, type]]]) -> List[Tuple[str, type]]:
        """
        Process and validate unique field definitions.

        Converts unique field specifications into a standardized format where each
        field is a tuple of (field_name, field_type). String fields can be specified
        as just the name and will default to str type.

        Parameters
        ----------
        fields : list
            List of field specifications. Each can be:
            - A string (field name, defaults to str type)
            - A tuple of (field_name, type) where type is a Python type class

        Returns
        -------
        list
            List of tuples, each containing (field_name, field_type).

        Raises
        ------
        ValueError
            If a field specification is not a string or a valid (string, type) tuple.
        """
        ufields = []
        for f in fields:
            if isinstance(f, tuple) and len(f) == 2 and type(f[1]) == type and isinstance(f[0], str):
                ufields.append(f)
            elif isinstance(f, str):
                ufields.append((f, str))
            else:
                raise ValueError(f"unique field {f} must be a string or a string-type tuple")
        return ufields

    def get_dbfile(self) -> Path:
        """
        Get the full path to the database file.

        Returns
        -------
        Path
            Path object pointing to the database file.
        """
        return Path(self.db_path) / (self.db_name + self.db_ext)

    def save_backup(self, return_out: bool = False) -> Optional[CompletedProcess]:
        """Save a backup of the database to the backup path specified in metadata.

        Parameters
        ----------
        return_out : bool, optional
            If True, return the output of the robocopy command.

        Returns
        -------
        CompletedProcess or None
            CompletedProcess object from the robocopy command (if return_out is True).
            Returns None if return_out is False.
        """
        source_path = self.db_path
        target_path = self.backup_path
        source_file = self.db_name + self.db_ext
        robocopy_arguments = f"robocopy {source_path} {target_path} {source_file}"
        outs = run(robocopy_arguments, capture_output=True, text=True)
        if return_out:
            return outs

    def connect(self) -> pyodbc.Connection:
        """
        Establish a connection to the database.

        Creates a pyodbc connection using the appropriate driver string based on
        the database file extension. Currently configured for Microsoft Access
        databases (.accdb, .mdb files).

        Returns
        -------
        pyodbc.Connection
            Database connection object.

        Raises
        ------
        AssertionError
            If the host type for the database extension is not supported.

        Notes
        -----
        To support additional database types:
        1. Determine the appropriate pyodbc driver string for your database
        2. Add it to the driver_string dictionary in this method
        3. Update the host_types dictionary at the module level to map your
           file extension to the driver key

        See https://www.connectionstrings.com/ for driver string examples.
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
    def open_cursor(self, commit_changes: bool = False) -> Generator[pyodbc.Cursor, None, None]:
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
            print(f"An exception occurred while trying to connect to {self.db_name}!")
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
    def show_metadata(self) -> None:
        """
        Display metadata associated with the open database.

        Prints information about the database location, name, table, unique ID field,
        backup path, and default filtering conditions.
        """
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

    def table_column_info(self) -> Tuple[List[str], List[str], List[bool]]:
        """
        Retrieve the column names, data types, and nullable status of the table.

        Returns
        -------
        tuple
            A tuple containing three elements:
            - A list of strings representing the column names of the table.
            - A list of strings representing the data types of the table.
            - A list of booleans representing the nullable status of the table.
        """
        with self.open_cursor(commit_changes=False) as cursor:
            query = f"SELECT * FROM {self.table_name} WHERE 1=0"
            cursor.execute(query)
            column_descriptions = cursor.description
        column_name, data_type, _, _, _, _, nullable = map(list, zip(*column_descriptions))
        return column_name, data_type, nullable

    # == retrieve table data ==
    def table_data(self) -> Tuple[List[str], List[Tuple[Any, ...]]]:
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
            field_names = [col.column_name for col in cursor.columns(table=self.table_name)]
            cursor.execute(f"SELECT * FROM {self.table_name}")
            table_elements = cursor.fetchall()

        return field_names, table_elements

    def get_table(self, use_default: bool = True, **kw_conditions: Any) -> pd.DataFrame:
        """
        Retrieve data from table in database and return as dataframe with optional filtering.

        This method retrieves all data from the primary table in the database specified in
        BaseDatabase instance. It automatically filters the data using the defaultConditions
        defined in the dbMetadata method. kw_conditions overwrite defaultConditions if there is
        a conflict.

        Parameters
        ----------
        use_default : bool, default=True
            Use default conditions if true, if False ignore them
        **kw_conditions : dict, optional
            Additional filtering conditions as keyword arguments.
            Each condition should match a column name in the table.
            Value can either be a variable (e.g. 0 or 'ATL000'), or a (value, operation) pair.
            The operation defaults to '==', but you can use anything that works as a df query.

            Examples:
                - Simple equality: ``imaging=True`` filters where imaging column equals True
                - Comparison operators: ``sessionID=(5, '>')`` filters where sessionID > 5
                - Multiple conditions: ``imaging=True, mouseName='ATL028'`` applies AND logic

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
        >>> df = vrdb.get_table(mouseName='ATL028', sessionID=(5, '>'))
        """

        field_names, table_data = self.table_data()
        df = pd.DataFrame.from_records(table_data, columns=field_names)
        conditions = copy(self.default_conditions) if use_default else {}
        conditions.update(kw_conditions)
        if conditions:
            for key, val in conditions.items():
                assert key in field_names, f"{key} is not a column name in {self.table_name}"
                conditions[key] = self.process_filter_value(val)  # make sure it's a value/operation pair
            query = " & ".join([self.construct_filter_string(key, val_op_tuple) for key, val_op_tuple in conditions.items()])
            df = df.query(query)
        return df

    def process_filter_value(self, val: Union[Any, Tuple[Any, str]]) -> Tuple[Any, str]:
        """
        Ensure filter value has an operation associated with it.

        Filters are passed to pandas DataFrame queries as {key}{operation}{value}.
        This method ensures each value is a tuple of (value, operation), defaulting
        to '==' if no operation is specified.

        Parameters
        ----------
        val : any or tuple
            Filter value. If a tuple, should be (value, operation). If not a tuple,
            will be converted to (val, '==').

        Returns
        -------
        tuple
            Tuple of (value, operation) for use in DataFrame queries.
        """
        if not isinstance(val, tuple):
            val = (val, "==")
        return val

    def construct_filter_string(self, key: str, val_op_tuple: Tuple[Any, str]) -> str:
        """
        Construct a string to be used as a pandas DataFrame query expression.

        Parameters
        ----------
        key : str
            Column name to filter on.
        val_op_tuple : tuple
            Tuple of (value, operation) where operation is a comparison operator
            (e.g., '==', '!=', '>', '<').

        Returns
        -------
        str
            Query string in the format `column_name`operator'value'.
        """
        val, op = val_op_tuple
        return f"`{key}`{op}{val!r}"

    # == methods for adding records and updating information to the database ==
    def create_update_statement(self, field: str, uid: Any) -> str:
        """
        Create an SQL UPDATE statement for a single field and record.

        Parameters
        ----------
        field : str
            Name of the field to update.
        uid : any
            Unique identifier value for the record to update.

        Returns
        -------
        str
            SQL UPDATE statement with parameter placeholder.

        Example
        -------
        >>> with self.open_cursor(commit_changes=True) as cursor:
        ...     cursor.execute(self.create_update_statement("fieldName", 123), value)
        """
        return f"UPDATE {self.table_name} set {field} = ? WHERE {self.uid} = {uid}"

    def create_update_many_statement(self, field: str) -> str:
        """
        Create an SQL UPDATE statement for batch updating a single field.

        Parameters
        ----------
        field : str
            Name of the field to update.

        Returns
        -------
        str
            SQL UPDATE statement with parameter placeholders for value and uid.

        Example
        -------
        >>> with self.open_cursor(commit_changes=True) as cursor:
        ...     stmt = self.create_update_many_statement("fieldName")
        ...     cursor.executemany(stmt, [(val1, uid1), (val2, uid2), ...])
        """
        return f"UPDATE {self.table_name} set {field} = ? where {self.uid} = ?"

    def update_database_field(self, field: str, val: Any, **kw_conditions: Any) -> None:
        """
        Update a database field for all records matching specified conditions.

        Parameters
        ----------
        field : str
            Name of the field to update.
        val : any
            Value to set for the field.
        **kw_conditions : dict, optional
            Filtering conditions to identify records to update.
            See get_table() documentation for filtering syntax.
            Examples: ``mouseName='ATL028'``, ``imaging=True``

        Raises
        ------
        AssertionError
            If the specified field is not in the database table.
        """
        assert field in self.table_data()[0], f"Requested field ({field}) is not in table. Use 'self.table_data()[0]' to see available fields."
        df = self.get_table(**kw_conditions)
        update_statement = self.create_update_many_statement(field)
        uids = df[self.uid].tolist()  # uids of all sessions requested
        val_as_list = [val] * len(uids)
        print(f"Setting {field}={val} for all requested records...")
        with self.open_cursor(commit_changes=True) as cursor:
            cursor.executemany(update_statement, zip(val_as_list, uids))

    # == method for adding a record to the database ==
    def add_record(self, insert_statement: str, columns: List[str], values: List[Any]) -> str:
        """
        Add a single record to the database.

        First checks if a record with matching unique field values already exists.
        If so, prevents duplicate insertion and returns a message. Otherwise,
        adds the new record to the database.

        Parameters
        ----------
        insert_statement : str
            SQL INSERT statement with parameter placeholders.
        columns : list
            List of column names matching the insert statement.
        values : list
            List of values to insert, corresponding to the columns.

        Returns
        -------
        str
            Success or duplicate record message.
        """
        d = dict(zip(columns, values))
        unique_values = [d[uf[0]] for uf in self.unique_fields]  # get values associated with unique fields
        for ii, uv in enumerate(unique_values):
            if isinstance(uv, date) or isinstance(uv, datetime):
                # this is required for communicating with Access
                unique_values[ii] = uv.strftime("%Y-%m-%d")
        unique_combo = ", ".join([f"{uf[0]}={uv}" for uf, uv in zip(self.unique_fields, unique_values)])
        if self.get_record(*unique_values, verbose=False) is not None:
            print(f"Record already exists for {unique_combo}")
            return f"Record already exists for {unique_combo}"
        with self.open_cursor(commit_changes=True) as cursor:
            cursor.execute(insert_statement, values)
            print(f"Successfully added new record for {unique_combo}")
        return "Successfully added new record"

    def get_record(self, *unique_values: Any, verbose: bool = True) -> Optional[pd.Series]:
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
        >>> record = vrdb.get_record(*unique_conditions)
        """

        # Check if correct values are provided
        if len(unique_values) != len(self.unique_fields):
            expected_list = ", ".join([uf[0] for uf in self.unique_fields])
            raise ValueError(f"{len(unique_values)} values provided but *get_record* is expecting values for: {expected_list}")

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
                unique_combo = ", ".join([f"{uf[0]}={uv}" for uf, uv in zip(self.unique_fields, unique_values)])
                print(f"No session found under: {unique_combo}")
            return None

        if len(df) > 1:
            unique_combo = ", ".join([f"{uf[0]}={uv}" for uf, uv in zip(self.unique_fields, unique_values)])
            raise ValueError(f"Multiple sessions found under: {unique_combo}")
        return df.iloc[0]


# ======== child database class definition for sessions ==========
class SessionDatabase(BaseDatabase):
    """
    Database class for handling VR session data.

    Specialized database class that extends BaseDatabase with session-specific
    functionality, including methods for creating session objects, managing
    registration workflows, and handling quality control processes.
    """

    def iter_sessions(self, session_params: Dict[str, Any] = {}, **kw_conditions: Any) -> List[B2Session]:
        """Iterate over sessions matching conditions.

        Parameters
        ----------
        session_params : dict, default={}
            Additional parameters to pass to the session constructor when creating
            B2Session objects. These are passed through to B2Session.create().
        **kw_conditions : dict, optional
            Additional filtering conditions passed to get_table().
            See get_table() documentation for filtering syntax.
            Examples: ``mouseName='ATL028'``, ``imaging=True``, ``sessionID=(5, '>')``

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
    def session_name(self, row: pd.Series) -> Tuple[str, str, str]:
        """
        Extract session identifiers from a database record.

        Parameters
        ----------
        row : pandas.Series
            Database record containing session information.

        Returns
        -------
        tuple
            Tuple of (mouse_name, session_date, session_id) where session_date is
            formatted as 'YYYY-MM-DD' and session_id is converted to string.
        """
        mouse_name = row["mouseName"]
        session_date = row["sessionDate"].strftime("%Y-%m-%d")
        session_id = str(row["sessionID"])
        return mouse_name, session_date, session_id

    def make_b2session(self, row: pd.Series) -> B2Session:
        """
        Create a B2Session object from a database record.

        Parameters
        ----------
        row : pandas.Series
            Database record containing session information.

        Returns
        -------
        B2Session
            Session object initialized with data from the record.
        """
        mouse_name, session_date, session_id = self.session_name(row)
        return B2Session.create(mouse_name, session_date, session_id)

    def make_b2registration(self, row: pd.Series, opts: B2RegistrationOpts) -> B2Registration:
        """
        Create a B2Registration object from a database record.

        Parameters
        ----------
        row : pandas.Series
            Database record containing session information.
        opts : B2RegistrationOpts
            Registration options to use for the session.

        Returns
        -------
        B2Registration
            Registration object initialized with session data and options.
        """
        mouse_name, session_date, session_id = self.session_name(row)
        return B2Registration(mouse_name, session_date, session_id, opts)

    # == helper functions for figuring out what needs work ==
    def needs_registration(self, skip_errors: bool = True, return_df: bool = True, **kw_conditions: Any) -> Union[pd.DataFrame, None]:
        """
        Get or print sessions that need registration preprocessing.

        Parameters
        ----------
        skip_errors : bool, default=True
            If True, exclude sessions that had registration errors.
        return_df : bool, default=True
            If True, returns a DataFrame. If False, prints the sessions instead.
        **kw_conditions : dict, optional
            Additional filtering conditions passed to get_table().
            See get_table() documentation for filtering syntax.

        Returns
        -------
        pandas.DataFrame or None
            If return_df=True, returns DataFrame containing sessions that need registration.
            If return_df=False, returns None and prints the sessions instead.
        """
        df = self.get_table(**kw_conditions)
        if skip_errors:
            df = df[df["vrRegistrationError"] == False]
        df = df[df["vrRegistration"] == False]

        if return_df:
            return df
        else:
            for idx, row in df.iterrows():
                session = self.make_b2session(row)
                print(f"Session needs registration: {session.session_print()}")
            return None

    def update_s2p_date_time(self) -> None:
        """
        Update suite2p creation dates in the database based on file modification times.

        For all sessions where suite2p processing is complete, finds the most recent
        file modification time in the suite2p output directory and updates the
        suite2pDate field in the database.
        """
        df = self.get_table()
        s2p_done = df[(df["imaging"] == True) & (df["suite2p"] == True)]
        uids = s2p_done[self.uid].tolist()
        s2p_creation_date = []
        for idx, row in s2p_done.iterrows():
            session = self.make_b2session(row)  # create vrSession to point to session folder
            c_latest_mod = 0
            for p in session.s2p_path.rglob("*"):
                if not (p.is_dir()):
                    c_latest_mod = max(p.stat().st_mtime, c_latest_mod)
            c_date_time = datetime.fromtimestamp(c_latest_mod)
            s2p_creation_date.append(c_date_time)  # get suite2p path creation date

        with self.open_cursor(commit_changes=True) as cursor:
            cursor.executemany(self.create_update_many_statement("suite2pDate"), zip(s2p_creation_date, uids))

    def needs_s2p(
        self, needs_qc: bool = False, return_df: bool = True, print_targets: bool = True, **kw_conditions: Any
    ) -> Union[pd.DataFrame, None]:
        """
        Get or print sessions that need suite2p processing or quality control.

        Parameters
        ----------
        needs_qc : bool, default=False
            If False, returns/prints sessions that need suite2p processing.
            If True, returns/prints sessions that need suite2p quality control.
        return_df : bool, default=True
            If True, returns a DataFrame. If False, prints the sessions instead.
        print_targets : bool, default=True
            If True and return_df=False, prints suite2p target information for sessions needing processing.
        **kw_conditions : dict, optional
            Additional filtering conditions passed to get_table().
            See get_table() documentation for filtering syntax.

        Returns
        -------
        pandas.DataFrame or None
            If return_df=True, returns DataFrame containing sessions that need suite2p processing or QC.
            If return_df=False, returns None and prints the sessions instead.
        """
        df = self.get_table(**kw_conditions)
        if needs_qc:
            df = df[(df["imaging"] == True) & (df["suite2p"] == True) & (df["suite2pQC"] == False)]
        else:
            df = df[(df["imaging"] == True) & (df["suite2p"] == False)]

        if return_df:
            return df
        else:
            for idx, row in df.iterrows():
                session = self.make_b2session(row)
                if needs_qc:
                    print(f"Database indicates that suite2p has been run but not QC'd: {session.session_print()}")
                else:
                    print(f"Database indicates that suite2p has not been run: {session.session_print()}")
                    if print_targets:
                        mouse_name, session_date, session_id = self.session_name(row)
                        s2p_targets(mouse_name, session_date, session_id)
                        print("")
            return None

    def check_s2p(self, with_database_update: bool = False, return_check: bool = False) -> Optional[bool]:
        """
        Verify suite2p status consistency between database and file system.

        Checks for discrepancies where:
        - Database says suite2p is done but files don't exist
        - Files exist but database says suite2p wasn't done

        Parameters
        ----------
        with_database_update : bool, default=False
            If True, automatically corrects database entries when discrepancies are found.
        return_check : bool, default=False
            If True, returns a boolean indicating whether any discrepancies were found.

        Returns
        -------
        bool or None
            If return_check is True, returns True if any discrepancies were found,
            False otherwise. Returns None if return_check is False.
        """
        df = self.get_table()

        # Check sessions where database says suite2p is done but files don't exist
        check_s2p_done = df[(df["imaging"] == True) & (df["suite2p"] == True)]
        checked_not_done = check_s2p_done.apply(lambda row: not (self.make_b2session(row).s2p_path.exists()), axis=1)
        not_actually_done = check_s2p_done[checked_not_done]

        # Check sessions where files exist but database says suite2p wasn't done
        check_s2p_needed = df[(df["imaging"] == True) & (df["suite2p"] == False)]
        checked_not_needed = check_s2p_needed.apply(lambda row: self.make_b2session(row).s2p_path.exists(), axis=1)
        not_actually_needed = check_s2p_needed[checked_not_needed]

        # Print database errors to workspace
        for idx, row in not_actually_done.iterrows():
            print(f"Database said suite2p has been ran, but it actually hasn't: {self.make_b2session(row).session_print()}")
        for idx, row in not_actually_needed.iterrows():
            print(f"Database said suite2p didn't run, but it already did: {self.make_b2session(row).session_print()}")

        # If with_database_update is True, correct the database
        if with_database_update:
            for idx, row in not_actually_done.iterrows():
                with self.open_cursor(commit_changes=True) as cursor:
                    cursor.execute(self.create_update_statement("suite2p", row[self.uid]), False)

            for idx, row in not_actually_needed.iterrows():
                with self.open_cursor(commit_changes=True) as cursor:
                    cursor.execute(self.create_update_statement("suite2p", row[self.uid]), True)

        # If return_check is requested, return True if any records were invalid
        if return_check:
            return checked_not_done.any() or checked_not_needed.any()

    # == for communicating with the database about red cell quality control ==
    def update_red_cell_qc_date_time(self) -> None:
        """
        Update red cell QC dates in the database based on file modification times.

        For all sessions where red cell QC is complete, finds the most recent
        modification time of relevant red cell QC files and updates the
        redCellQCDate field in the database.
        """
        relevant_one_files = [
            "mpciROIs.redCellIdx.npy",
            "mpciROIs.redCellManualAssignment.npy",
            "parametersRed*",  # wildcard because there are multiple possibilities
        ]

        df = self.get_table()
        red_cell_qc_done = df[(df["imaging"] == True) & (df["suite2p"] == True) & (df["redCellQC"] == True)]
        uids = red_cell_qc_done[self.uid].tolist()
        rc_edit_date = []
        for idx, row in red_cell_qc_done.iterrows():
            session = self.make_b2session(row)  # create vrSession to point to session folder
            c_latest_mod = 0
            for f in relevant_one_files:
                for file in session.one_path.rglob(f):
                    c_latest_mod = max(file.stat().st_mtime, c_latest_mod)
            c_date_time = datetime.fromtimestamp(c_latest_mod)
            rc_edit_date.append(c_date_time)  # get red cell QC file modification date

        with self.open_cursor(commit_changes=True) as cursor:
            cursor.executemany(self.create_update_many_statement("redCellQCDate"), zip(rc_edit_date, uids))

    def needs_red_cell_qc(self, return_df: bool = True, **kw_conditions: Any) -> Union[pd.DataFrame, None]:
        """
        Get or print sessions that need red cell quality control.

        Parameters
        ----------
        return_df : bool, default=True
            If True, returns a DataFrame. If False, prints the sessions instead.
        **kw_conditions : dict, optional
            Additional filtering conditions passed to get_table().
            See get_table() documentation for filtering syntax.

        Returns
        -------
        pandas.DataFrame or None
            If return_df=True, returns DataFrame containing sessions that need red cell QC.
            Sessions must have imaging, suite2p processing, and registration completed.
            If return_df=False, returns None and prints the sessions instead.
        """
        df = self.get_table(**kw_conditions)
        df = df[(df["imaging"] == True) & (df["suite2p"] == True) & (df["vrRegistration"] == True) & (df["redCellQC"] == False)]

        if return_df:
            return df
        else:
            for idx, row in df.iterrows():
                print(f"Database indicates that redCellQC has not been performed for session: {self.make_b2session(row).session_print()}")
            return None

    def iter_sessions_need_red_cell_qc(self, **kw_conditions: Any) -> List[B2Session]:
        """
        Get list of sessions that require red cell quality control.

        Parameters
        ----------
        **kw_conditions : dict, optional
            Additional filtering conditions passed to get_table().
            See get_table() documentation for filtering syntax.

        Returns
        -------
        list[B2Session]
            List of session objects that need red cell QC.
        """
        df = self.needs_red_cell_qc(return_df=True, **kw_conditions)
        sessions = []
        for idx, row in df.iterrows():
            sessions.append(self.make_b2session(row))
        return sessions

    def set_red_cell_qc(self, mouse_name: str, date_string: str, session_id: Union[str, int], state: bool = True) -> bool:
        """
        Set the red cell QC status for a specific session.

        Parameters
        ----------
        mouse_name : str
            Mouse name identifier.
        date_string : str
            Session date in 'YYYY-MM-DD' format.
        session_id : str or int
            Session ID.
        state : bool, default=True
            Red cell QC status to set. If True, also sets the QC date to now.

        Returns
        -------
        bool
            True if update was successful, False otherwise.
        """
        record = self.get_record(mouse_name, date_string, session_id)
        if record is None:
            print(f"Could not find session {mouse_name}/{date_string}/{session_id} in database.")
            return False

        try:
            with self.open_cursor(commit_changes=True) as cursor:
                cursor.execute(self.create_update_statement("redCellQC", record[self.uid]), state)
                if state == True:
                    # If setting red cell QC to true, add the date
                    cursor.execute(
                        self.create_update_statement("redCellQCDate", record[self.uid]),
                        datetime.now(),
                    )
                else:
                    # Otherwise remove the date
                    cursor.execute(self.create_update_statement("redCellQCDate", record[self.uid]), "")
            return True

        except Exception as ex:
            print(f"Failed to update database for session: {mouse_name}/{date_string}/{session_id}")
            print(f"Error: {ex}")
            return False

    # == operating vrExperiment pipeline ==
    def register_record(self, record: pd.Series, raise_exception: bool = False, imaging: Optional[bool] = None) -> Tuple[bool, int]:
        """
        Perform registration preprocessing for a single session record.

        Creates a B2Registration object and runs preprocessing. Updates the database
        with success/failure status and error information if applicable.

        Parameters
        ----------
        record : pandas.Series
            Database record containing session information.
        raise_exception : bool, default=False
            If True, raises exceptions instead of handling them silently.
        imaging : bool, optional
            Override the imaging setting. If None (default), uses the value from the
            database record. If True or False, overrides the database value.

        Returns
        -------
        tuple
            Tuple of (success, data_size) where:
            - success: bool indicating if registration succeeded
            - data_size: int size in bytes of registered oneData (0 if failed)

        Notes
        -----
        On failure, clears all oneData files and updates database error fields.
        On success, updates registration status and date in the database.
        """
        opts = B2RegistrationOpts()
        opts.imaging = bool(imaging) if imaging is not None else bool(record["imaging"])
        opts.facecam = bool(record["faceCamera"])
        opts.vrBehaviorVersion = record["vrBehaviorVersion"]
        b2reg = self.make_b2registration(record, opts)
        try:
            print(f"Registering data for session: {b2reg.session_print()}")
            b2reg.register()
        except Exception as ex:
            with self.open_cursor(commit_changes=True) as cursor:
                cursor.execute(self.create_update_statement("vrRegistrationError", record[self.uid]), True)
                cursor.execute(
                    self.create_update_statement("vrRegistrationException", record[self.uid]),
                    str(ex),
                )
            if raise_exception:
                raise ex
            print(f"The following exception was raised when trying to preprocess session: {b2reg.session_print()}. Clearing all oneData.")
            b2reg.clear_one_data(certainty=True)
            error_print(f"Last traceback: {traceback.extract_tb(ex.__traceback__, limit=-1)}")
            error_print(f"Exception: {ex}")
            # If failed, return (False, 0)
            out = (False, 0)
        else:
            with self.open_cursor(commit_changes=True) as cursor:
                # Tell the database that vrRegistration was performed and the time of processing
                cursor.execute(self.create_update_statement("vrRegistration", record[self.uid]), True)
                cursor.execute(self.create_update_statement("vrRegistrationError", record[self.uid]), False)
                cursor.execute(self.create_update_statement("vrRegistrationException", record[self.uid]), "")
                cursor.execute(
                    self.create_update_statement("vrRegistrationDate", record[self.uid]),
                    datetime.now(),
                )
            # If successful, return (True, size of registered oneData)
            out = (True, sum([one_file.stat().st_size for one_file in b2reg.get_saved_one()]))
            print(f"Session {b2reg.session_print()} registered with {readable_bytes(out[1])} oneData.")
        finally:
            del b2reg
        return out

    def register_single_session(
        self,
        mouse_name: str,
        session_date: str,
        session_id: Union[str, int],
        raise_exception: bool = False,
        imaging: Optional[bool] = None,
    ) -> Optional[bool]:
        """
        Register a single session by its identifiers.

        Parameters
        ----------
        mouse_name : str
            Mouse name identifier.
        session_date : str
            Session date in 'YYYY-MM-DD' format.
        session_id : str or int
            Session ID.
        raise_exception : bool, default=False
            If True, raises exceptions instead of handling them silently.
        imaging : bool, optional
            Override the imaging setting. If None (default), uses the value from the
            database record. If True or False, overrides the database value to enable
            or disable imaging processing during registration.

        Returns
        -------
        bool or None
            True if registration succeeded, False if failed, None if session not found.
        """
        record = self.get_record(mouse_name, session_date, session_id)
        if record is None:
            print(f"Session {'/'.join([mouse_name, session_date, session_id])} is not in the database")
            return
        out = self.register_record(record, raise_exception=raise_exception, imaging=imaging)
        return out[0]

    def register_sessions(
        self,
        max_data: float = 30e9,
        skip_errors: bool = True,
        raise_exception: bool = False,
        imaging: Optional[bool] = None,
    ) -> None:
        """
        Register multiple sessions that need registration.

        Processes sessions in batches, stopping when the total data size limit is reached.
        Provides progress updates including accumulated data size and estimates.

        Parameters
        ----------
        max_data : float, default=30e9
            Maximum total data size (in bytes) to process before stopping.
            Default is 30 GB.
        skip_errors : bool, default=True
            If True, skip sessions that had previous registration errors.
        raise_exception : bool, default=False
            If True, raises exceptions instead of handling them silently.
        imaging : bool, optional
            Override the imaging setting for all sessions. If None (default), uses the
            value from each session's database record. If True or False, overrides the
            database value for all sessions to enable or disable imaging processing.

        Notes
        -----
        Prints progress information including:
        - Accumulated oneData registered
        - Average data size per session
        - Estimated remaining data to process
        """
        count_sessions = 0
        total_one_data = 0.0
        df_to_register = self.needs_registration(skip_errors=skip_errors)

        for idx, (_, row) in enumerate(df_to_register.iterrows()):
            if total_one_data > max_data:
                print(f"\nMax data limit reached. Total processed: {readable_bytes(total_one_data)}. Limit: {readable_bytes(max_data)}")
                return
            print("")
            out = self.register_record(row, raise_exception=raise_exception, imaging=imaging)
            if out[0]:
                count_sessions += 1  # count successful sessions
                total_one_data += out[1]  # accumulated oneData registered
                estimate_remaining = len(df_to_register) - idx - 1
                print(
                    f"Accumulated oneData registered: {readable_bytes(total_one_data)}. "
                    f"Averaging: {readable_bytes(total_one_data/count_sessions)} / session. "
                    f"Estimate remaining: {readable_bytes(total_one_data/count_sessions*estimate_remaining)}"
                )

    def print_registration_errors(self, **kw_conditions: Any) -> None:
        """
        Print registration errors for sessions that failed registration.

        Parameters
        ----------
        **kw_conditions : dict, optional
            Additional filtering conditions passed to get_table().
            See get_table() documentation for filtering syntax.
        """
        df = self.get_table(**kw_conditions)
        for idx, row in df[df["vrRegistrationError"] == True].iterrows():
            print(f"{'/'.join(self.session_name(row))} had error: {row['vrRegistrationException']}")
