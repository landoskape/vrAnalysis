# Database Management

The `vrAnalysis2.database` module provides classes and functions for managing VR session data in a database. It's designed to work with Microsoft Access databases (`.accdb` files) by default, but can be adapted to other SQL databases.

## Core Classes

### BaseDatabase

The base class for database operations. Provides core functionality for connecting to databases, querying records, and updating data.

**Key Methods:**

- `get_table()`: Query records from the database table
- `get_record()`: Retrieve a single record by unique identifiers
- `update_database_field()`: Update field values for matching records
- `add_record()`: Add new records to the database

**Example:**

```python
from vrAnalysis2.database import get_database

# Get a database instance
db = get_database("vrMice")

# Query records
mice = db.get_table()

# Get a specific record
mouse = db.get_record("mouse001")

# Update a field
db.update_database_field("someField", "newValue", mouseName="mouse001")
```

### SessionDatabase

Specialized database class for managing VR sessions. Extends `BaseDatabase` with session-specific functionality.

**Key Methods:**

- `iter_sessions()`: Create B2Session objects from database records
- `make_b2session()`: Create a B2Session from a database row
- `make_b2registration()`: Create a B2Registration object for preprocessing
- `needs_registration()`: Find sessions that need registration
- `needs_s2p()`: Find sessions that need suite2p processing or QC
- `check_s2p()`: Verify suite2p status consistency

**Example:**

```python
from vrAnalysis2.database import get_database

# Get session database
db = get_database("vrSessions")

# Find sessions needing registration
needs_reg = db.needs_registration(mouseName="mouse001")

# Create session objects
sessions = db.iter_sessions(mouseName="mouse001", sessionQC=True)

# Check suite2p status
db.check_s2p(with_database_update=True)
```

## Configuration

Database configuration is managed through the `get_database_metadata()` function. You'll need to edit this function to match your database setup:

```python
def get_database_metadata(db_name: str) -> dict:
    dbdict = {
        "vrSessions": {
            "db_path": r"C:\path\to\your\database",
            "db_name": "vrDatabase",
            "db_ext": ".accdb",
            "table_name": "sessiondb",
            "uid": "uSessionID",
            "backup_path": r"D:\backup\path",
            "unique_fields": [("mouseName", str), ("sessionDate", datetime), ("sessionID", int)],
            "default_conditions": {"sessionQC": True},
            "constructor": SessionDatabase,
        },
        # ... other databases
    }
    return dbdict[db_name]
```

## Querying Data

The `get_table()` method supports flexible querying:

```python
# Simple equality
df = db.get_table(mouseName="mouse001")

# Comparison operators
df = db.get_table(sessionID=(5, ">"))  # sessionID > 5

# Multiple conditions (AND logic)
df = db.get_table(mouseName="mouse001", imaging=True)

# Disable default conditions
df = db.get_table(use_default=False, mouseName="mouse001")
```

## Adding Records

Use the GUI to add new records:

```python
from vrAnalysis2.uilib.add_entry_gui import add_entry_gui

# Open GUI for adding entries
add_entry_gui("vrSessions")
```

Or programmatically:

```python
# Create insert statement
columns = ["mouseName", "sessionDate", "sessionID", ...]
values = ["mouse001", datetime(2024, 1, 15), 1, ...]
insert_stmt = f"INSERT INTO {db.table_name} ({', '.join(columns)}) VALUES ({', '.join(['?'] * len(columns))})"

# Add record
db.add_record(insert_stmt, columns, values)
```

## Database Backup

Automatically backup your database:

```python
db.save_backup()
```

## See Also

- [Quickstart Guide](../quickstart.md) for basic usage
- [API Reference](../api/database.md) for complete function signatures

