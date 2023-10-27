import sys
from copy import copy
import math
import importlib
from functools import partial
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QTextEdit, QPushButton, QVBoxLayout, QHBoxLayout, QCheckBox, QFormLayout, QDateEdit
from PyQt5.QtCore import Qt, QRegExp, QDate

import random
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
list_ignore = ['uSessionID']

# default values
default_vals = {
    'vrRegistration': False,
    'suite2p': False, 
    'suite2pQC': False, 
    'redCellQC': False,
    'sessionQC': True,
    'vrRegistrationError': False,
}

# default parameters
params = {
    'max_rows' : 12,
    'min_width' : 240,
}


def get_column_descriptions(vrdb):
    with vrdb.openCursor(commitChanges=False) as cursor:
        query = f"SELECT * FROM {vrdb.tableName} WHERE 1=0"
        cursor.execute(query)
        column_descriptions = cursor.description
    return column_descriptions

class newEntryGUI(QWidget):
    def __init__(self, vrdb, ses=None, **kwargs):
        super().__init__()
        column_descriptions = get_column_descriptions(vrdb)
        column_name, data_type, size, _, _, _, nullable = map(list, zip(*column_descriptions))
        self.vrdb = vrdb
        self.column_name = column_name
        self.data_type = data_type
        self.size = size
        self.nullable = nullable
        self.defaults = [default_vals[cname] if cname in default_vals else None for cname in column_name]
            
        # GUI parameters
        self.params = copy(params)
        self.params.update(kwargs)

        # Basic second-order variables
        self.num_columns = len(column_name) # this is the number of columns in the database
        
        # Create GUI
        self.init_ui()

        # And open it
        self.show()
        
    def init_ui(self):
        # First, check and select which columns require the user to provide information
        ignore_column = [cname in list_ignore for cname in self.column_name]
        self.num_required = sum([not(ic) for ic in ignore_column])
        self.gui_columns = math.ceil(self.num_required / self.params['max_rows'])

        # For each column
        # For each column name, create a label and an edit field
        self.entryLabels = []
        self.entryFields = []
        self.entryIndex = []
        for idx, (ignore, name, dtype, ss, null) in enumerate(zip(ignore_column, self.column_name, self.data_type, self.size, self.nullable)):
            if ignore: continue

            # otherwise, make a label and an edit field
            self.entryLabels.append(QLabel(f"{name}:")) # name the entry
            self.entryFields.append(QLineEdit(placeholderText=self.constructPlaceholder(idx))) # create the edit field
            self.entryIndex.append(idx) # keep track of which ones were not ignored

        # Set validation functions for each field and control their appearance
        for idx, (field, entry) in enumerate(zip(self.entryFields, self.entryIndex)):
            field.editingFinished.connect(partial(self.validate_input, idx=idx, entry=entry))
            field.setMinimumWidth(self.params['min_width'])
            
        # Create a button for submitting the new entry to a table
        self.checkEntry = QPushButton("Check Values")
        self.checkEntry.clicked.connect(self.checkValues)
        self.submitEntry = QPushButton("Submit New Row")
        self.submitEntry.clicked.connect(self.addNewEntry)
        
        # Create the layout and add widgets
        full_layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        
        # This is the entry edit fields, organized into a few columns
        col_layout = QHBoxLayout()
        subcol_layouts = [QFormLayout() for _ in range(self.gui_columns)]
        for ii, (label, field) in enumerate(zip(self.entryLabels, self.entryFields)):
            ccol = ii // self.params['max_rows']
            crow = ii - ccol
            subcol_layouts[ccol].addRow(label, field)
            
        for clay in subcol_layouts:
            col_layout.addLayout(clay)

        button_layout.addWidget(self.checkEntry)
        button_layout.addWidget(self.submitEntry)
        
        # Then put everything together in rows, 
        full_layout.addLayout(col_layout)
        full_layout.addLayout(button_layout)
        
        # Set the main layout for the window
        self.setLayout(full_layout)

        # Set the window properties
        self.setWindowTitle("Add Entry to Database")
        self.setGeometry(100, 100, 400, 300)

        # Apply custom CSS styling (optional - if you want to keep the styling from the previous step)
        self.setStyleSheet(darkModeStylesheet)

    def validate_input(self, event=None, idx=-1, entry=-1, return_value=False):
        assert idx != -1, "invalid index was provided"
        assert entry != -1, "invalid entry index was provided"
        valid = True # initialize to true
        ctext = self.entryFields[idx].text()
        value = None
        has_default = (self.defaults[entry] is not None)
        null = (ctext == '')
        
        # if not nullable and doesn't have default and data is null, not valid
        if not(self.nullable[entry]) and not(has_default) and null:
            valid = False
            value = None

        # insert default value if exists 
        if has_default and null:
            value = self.defaults[entry]
        
        # if data is empty and it is either nullable or has a default, ignore other checks 
        ignore_checks = (self.nullable[entry] or has_default) and null
        
        # if int and data is not digits, then not valid
        if self.data_type[entry] == int and not(ignore_checks):
            if not ctext.isdigit():
                valid = False
                value = None
            else:
                value = int(ctext)

        if self.data_type[entry] == float and not(ignore_checks):
            try:
                value = float(ctext)
            except:
                valid = False
                value = None
                
        # if datetime and can't make input into datetime, then not valid
        if self.data_type[entry] == datetime and not(ignore_checks):
            try:
                value = datetime.strptime(ctext, '%Y-%m-%d')  # Change the date format as needed
            except:
                valid = False
                value = None
                
        # if bool, then must be "True" or "False" or "1" or "0" 
        if self.data_type[entry] == bool and not(ignore_checks):
            valid = ctext == 'True' or ctext == 'False' or ctext == '1' or ctext == '0'    
            if valid and (ctext=='True' or ctext=='1'):
                value = True
            if valid and (ctext=='False' or ctext=='0'):
                value = False
            if not(valid):
                value = None
            
        # if string, just check that it isn't null
        if self.data_type[entry] == str and not(ignore_checks):
            valid = not(null)
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

    def constructPlaceholder(self, entry):
        placeholder = str(self.data_type[entry])
        if self.nullable[entry]:
            placeholder += ' (nullable)'
        if self.size[entry] is not None:
            placeholder += f' max characters: {self.size[entry]}'
        if self.defaults[entry] is not None:
            placeholder += f' default={self.defaults[entry]}'
        return placeholder
        
    def checkValidity(self):
        validEntries = [self.validate_input(idx=idx, entry=entry) for idx, entry in enumerate(self.entryIndex)]
        return all(validEntries)
    
    def checkValues(self):
        validData = self.checkValidity()
        if validData:
            self.outputEntry.setPlainText("Values are all valid! Ready to submit")
        else:
            self.outputEntry.setPlainText("Some fields do not have valid input data!")

    def get_values(self):
        print("Note this isn't smart with already calling checkValidity()")
        columns = []
        values = []
        for idx, entry in enumerate(self.entryIndex):
            columns.append(self.column_name[entry])
            values.append(self.validate_input(idx=idx, entry=entry, return_value=True)[1])          
        return columns, values
        
    def get_insert_statement(self, columns, values):
        assert len(columns)==len(values), "columns and values have a different length!"
        column_names = ", ".join(columns)
        value_holders = ", ".join(["?"] * len(columns))
        insert_statement = f"INSERT INTO {self.vrdb.tableName} ({column_names}) VALUES ({value_holders});"
        return insert_statement
        
    def addNewEntry(self):
        validData = self.checkValidity()
        columns, values = self.get_values()
        insert_statement = self.get_insert_statement(columns, values)
        
        if validData:
            self.vrdb.addRecord(insert_statement, values)
            print("Submission successful")
        else:
            print("Some fields do not have valid input data, submission failed!")









