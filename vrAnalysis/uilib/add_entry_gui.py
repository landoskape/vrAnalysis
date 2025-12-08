import sys
from copy import copy
import math
from functools import partial
from typing import Optional, List, Tuple, Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..database import BaseDatabase

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QTextEdit,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
)
from PyQt5.QtCore import Qt

from datetime import datetime

# prepare GUI
darkModeStylesheet = """
    QWidget {
        background-color: #1F1F1F;
        color: #F0F0F0;
        font-family: Arial, sans-serif;
    }

    QLabel {
        color: #E0E0E0;
        font-size: 14px;
        font-weight: bold;
    }

    QLineEdit, QTextEdit, QDateEdit {
        background-color: #333333;
        color: #F0F0F0;
        font-size: 13px;
        border: 1px solid #555555;
        border-radius: 5px;
        padding: 5px;
    }

    QLineEdit:focus, QTextEdit:focus, QDateEdit:focus {
        border: 1px solid #7F7F7F;
    }

    QPushButton {
        background-color: #4CAF50;
        color: #F0F0F0;
        font-size: 14px;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
    }

    QPushButton:hover {
        background-color: #45a049;
        font-size: 14px;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
    }

    QTextEdit[readOnly="true"] {
        background-color: #1F1F1F;
        border: none;
    }
"""

darkModeErrorStyle = "background-color: #CC6666;"

# table elements to ignore
# this has to be hard coded because I don't know how to do it programmatically :(
list_ignore = ["uSessionID"]

# default values
default_vals = {
    "vrRegistration": False,
    "suite2p": False,
    "suite2pQC": False,
    "redCellQC": False,
    "sessionQC": True,
    "vrRegistrationError": False,
    "dontTrack": False,
}

# recognized datatypes
recognized_types = [int, str, datetime, float, bool]

# import from session attributes
# Maps database column names to B2Session attribute names
attribute_lookup = {
    "mouseName": "mouse_name",
    "sessionDate": "date",
    "sessionID": "session_id",
}

# default parameters
params = {
    "max_rows": 10,
    "min_width": 240,
    "label_width": 190,
}


class NewEntryGUI(QWidget):
    """
    GUI for adding new entries to a database table.

    This widget provides a form-based interface for entering data into database
    fields with validation and automatic population from session objects.
    """

    def __init__(self, vrdb: "BaseDatabase", ses: Optional[Any] = None, **kwargs):
        # Ensure QApplication exists before creating widgets
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        super().__init__()

        try:
            column_name, data_type, nullable = vrdb.table_column_info()
        except Exception as e:
            print(f"Error getting column descriptions: {e}")
            print("Cannot create GUI without database schema information.")
            return

        self.vrdb = vrdb
        self.ses = ses
        self.column_name = column_name
        self.data_type = data_type
        self.nullable = nullable
        self.defaults = [default_vals[cname] if cname in default_vals else None for cname in column_name]

        # GUI parameters
        self.params = copy(params)
        self.params.update(kwargs)

        # Basic second-order variables
        self.num_columns = len(column_name)  # this is the number of columns in the database

        # Create GUI
        print(f"Creating GUI for database table: {self.vrdb.table_name}")
        print(f"Found {len(column_name)} columns in database")
        self.init_ui()

        # If session object is provided, fill in details
        self.populate_from_session()

        # If kwargs provided, add them to GUI
        self.populate_from_kwargs(**kwargs)

        # And open it - with special handling for Jupyter notebooks
        print("Showing GUI window...")
        self.show()
        self.raise_()  # Bring window to front
        self.activateWindow()  # Activate the window

        # For Jupyter notebooks, we might need to process events
        if app:
            app.processEvents()

        print(f"GUI window created and shown. Window visible: {self.isVisible()}")

    def populate_from_kwargs(self, **kwargs) -> None:
        """
        Populate form fields from keyword arguments.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments matching column names. Values are set in the
            corresponding form fields.
        """
        for idx, entry in enumerate(self.entryIndex):
            if self.column_name[entry] in kwargs:
                self.entryFields[idx].setText(str(kwargs[self.column_name[entry]]))

    def populate_from_session(self) -> None:
        """
        Populate form fields from session object attributes.

        Uses the attribute_lookup dictionary to map database column names
        to session object attribute names (e.g., B2Session attributes).
        """
        if self.ses is None:
            return  # nothing to do if not provided
        for idx, entry in enumerate(self.entryIndex):
            # check if column name matches attribute look up table
            if self.column_name[entry] in attribute_lookup:
                attribute = attribute_lookup[self.column_name[entry]]
                # if session object has the requested attribute, then...
                if hasattr(self.ses, attribute):
                    # set the value in the LineEdit
                    self.entryFields[idx].setText(getattr(self.ses, attribute))

    def init_ui(self) -> None:
        """
        Initialize the user interface components.

        Creates form fields for all database columns (except ignored ones),
        sets up validation, and arranges widgets in a layout.
        """
        # First, check and select which columns require the user to provide information
        ignore_column = [cname in list_ignore for cname in self.column_name]
        self.num_required = sum([not (ic) for ic in ignore_column])
        self.gui_columns = math.ceil(self.num_required / self.params["max_rows"])

        # For each column
        # For each column name, create a label and an edit field
        self.entryLabels = []
        self.entryFields = []
        self.entryIndex = []
        for idx, (ignore, name, dtype, null) in enumerate(zip(ignore_column, self.column_name, self.data_type, self.nullable)):
            if ignore:
                continue

            # otherwise, make a label and an edit field
            self.entryLabels.append(QLabel(f"{name}:"))  # name the entry
            self.entryFields.append(QLineEdit(placeholderText=self.construct_placeholder(idx)))  # create the edit field
            self.entryIndex.append(idx)  # keep track of which ones were not ignored

        for label in self.entryLabels:
            label.setFixedWidth(self.params["label_width"])
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # Set validation functions for each field and control their appearance
        for idx, (field, entry) in enumerate(zip(self.entryFields, self.entryIndex)):
            field.editingFinished.connect(partial(self.validate_input, idx=idx, entry=entry))
            field.setMinimumWidth(self.params["min_width"])

        # Create output text area for status messages
        self.outputEntry = QTextEdit()
        self.outputEntry.setReadOnly(True)
        self.outputEntry.setMaximumHeight(80)
        self.outputEntry.setPlainText("Ready to input values...")

        # Create a button for submitting the new entry to a table
        self.checkEntry = QPushButton("Check Values")
        self.checkEntry.clicked.connect(self.check_values)
        self.submitEntry = QPushButton("Submit New Row")
        self.submitEntry.clicked.connect(self.add_new_entry)

        # Create the layout and add widgets
        full_layout = QVBoxLayout()
        button_layout = QHBoxLayout()

        # This is the entry edit fields, organized into a few columns
        col_layout = QHBoxLayout()
        subcol_layouts = [QFormLayout() for _ in range(self.gui_columns)]
        for ii, (label, field) in enumerate(zip(self.entryLabels, self.entryFields)):
            ccol = ii // self.params["max_rows"]
            crow = ii - ccol
            subcol_layouts[ccol].addRow(label, field)

        for clay in subcol_layouts:
            col_layout.addLayout(clay)

        button_layout.addWidget(self.checkEntry)
        button_layout.addWidget(self.submitEntry)

        # Then put everything together in rows,
        full_layout.addLayout(col_layout)
        full_layout.addWidget(self.outputEntry)
        full_layout.addLayout(button_layout)

        # Set the main layout for the window
        self.setLayout(full_layout)

        # Set the window properties
        self.setWindowTitle("Add Entry to Database")
        self.setGeometry(100, 100, 400, 300)

        # Apply custom CSS styling (optional - if you want to keep the styling from the previous step)
        self.setStyleSheet(darkModeStylesheet)

    def validate_input(self, event=None, idx: int = -1, entry: int = -1, return_value: bool = False) -> Union[bool, Tuple[bool, Any]]:
        """
        Validate input for a specific form field.

        Parameters
        ----------
        event : optional
            Event object (not used, kept for compatibility with signal handlers).
        idx : int
            Index in entryFields list.
        entry : int
            Index in entryIndex list (corresponds to database column index).
        return_value : bool, default=False
            If True, returns tuple of (valid, value). If False, returns only valid.

        Returns
        -------
        bool or tuple[bool, any]
            If return_value is False, returns bool indicating validity.
            If return_value is True, returns tuple of (valid, value).
        """
        assert idx != -1, "invalid index was provided"
        assert entry != -1, "invalid entry index was provided"
        valid = True  # initialize to true
        ctext = self.entryFields[idx].text()
        value = None
        has_default = self.defaults[entry] is not None
        null = ctext == ""

        assert (
            self.data_type[entry] in recognized_types
        ), f"Datatype {self.data_type} is not registered, add it to the list of recognized types and update validate_input!"

        # if not nullable and doesn't have default and data is null, not valid
        if not (self.nullable[entry]) and not (has_default) and null:
            valid = False
            value = None

        # insert default value if exists
        if has_default and null:
            value = self.defaults[entry]

        # if data is empty and it is either nullable or has a default, ignore other checks
        ignore_checks = (self.nullable[entry] or has_default) and null

        # if int and data is not digits, then not valid
        if self.data_type[entry] == int and not (ignore_checks):
            try:
                value = int(ctext)
            except ValueError:
                valid = False
                value = None

        if self.data_type[entry] == float and not (ignore_checks):
            try:
                value = float(ctext)
            except:
                valid = False
                value = None

        # if datetime and can't make input into datetime, then not valid
        if self.data_type[entry] == datetime and not (ignore_checks):
            try:
                value = datetime.strptime(ctext, "%Y-%m-%d")  # Change the date format as needed
            except:
                valid = False
                value = None

        # if bool, then must be "True" or "False" or "1" or "0"
        if self.data_type[entry] == bool and not (ignore_checks):
            valid = ctext == "True" or ctext == "False" or ctext == "1" or ctext == "0"
            if valid and (ctext == "True" or ctext == "1"):
                value = True
            if valid and (ctext == "False" or ctext == "0"):
                value = False
            if not (valid):
                value = None

        # if string, just check that it isn't null
        if self.data_type[entry] == str and not (ignore_checks):
            valid = not (null)
            value = ctext if valid else None

        if valid:
            self.entryFields[idx].setStyleSheet("")
            if return_value:
                return True, value
            else:
                return True
        else:
            self.entryFields[idx].setStyleSheet(darkModeErrorStyle)
            if return_value:
                return False, value
            else:
                return False

    def construct_placeholder(self, entry: int) -> str:
        """
        Construct a placeholder text for a form field.

        Parameters
        ----------
        entry : int
            Index of the entry in entryIndex.

        Returns
        -------
        str
            Placeholder text describing the field's data type and constraints.
        """
        placeholder = str(self.data_type[entry])
        if self.nullable[entry]:
            placeholder += " (nullable)"
        if self.defaults[entry] is not None:
            placeholder += f" default={self.defaults[entry]}"
        return placeholder

    def check_validity(self) -> bool:
        """
        Check if all input fields contain valid data.

        Returns
        -------
        bool
            True if all fields are valid, False otherwise.
        """
        valid_entries = [self.validate_input(idx=idx, entry=entry) for idx, entry in enumerate(self.entryIndex)]
        return all(valid_entries)

    def check_values(self) -> None:
        """Check field values and display validation status in the output area."""
        valid_data = self.check_validity()
        if valid_data:
            self.outputEntry.setPlainText("Values are all valid! Ready to submit")
        else:
            self.outputEntry.setPlainText("Some fields do not have valid input data!")

    def get_values(self) -> Tuple[List[str], List[Any]]:
        """
        Extract validated values from all form fields.

        Returns
        -------
        tuple[list[str], list[any]]
            Tuple of (column_names, values) ready for database insertion.
        """
        print("This could be coded smarter with already calling check_validity()")
        columns = []
        values = []
        for idx, entry in enumerate(self.entryIndex):
            columns.append(self.column_name[entry])
            values.append(self.validate_input(idx=idx, entry=entry, return_value=True)[1])
        return columns, values

    def get_insert_statement(self, columns: List[str], values: List[Any]) -> str:
        """
        Generate an SQL INSERT statement for the given columns and values.

        Parameters
        ----------
        columns : list[str]
            List of column names.
        values : list[any]
            List of values corresponding to the columns.

        Returns
        -------
        str
            SQL INSERT statement with parameter placeholders.
        """
        assert len(columns) == len(values), "columns and values have a different length!"
        column_names = ", ".join(columns)
        value_holders = ", ".join(["?"] * len(columns))
        insert_statement = f"INSERT INTO {self.vrdb.table_name} ({column_names}) VALUES ({value_holders});"
        return insert_statement

    def add_new_entry(self) -> None:
        """
        Validate and submit a new entry to the database.

        Validates all fields, then attempts to insert the record into the database.
        Updates the output area with success or error messages.
        """
        valid_data = self.check_validity()
        columns, values = self.get_values()
        insert_statement = self.get_insert_statement(columns, values)

        if valid_data:
            try:
                output = self.vrdb.add_record(insert_statement, columns, values)
                self.outputEntry.setPlainText("Successfully added new entry to database!")
            except Exception as e:
                self.outputEntry.setPlainText(f"Error adding entry to database: {e}")
                print(f"Database error: {e}")
        else:
            error_msg = "Some fields do not have valid input data, submission failed!"
            self.outputEntry.setPlainText(error_msg)
            print(error_msg)
