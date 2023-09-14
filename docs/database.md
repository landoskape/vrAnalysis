# vrAnalysis: database

This is some documentation for the database of vrAnalysis. The main goal of
this file is to remind me what I've written so I don't forget. There may be 
room for explaining how it works, but at the moment, I'll just focus on using 
it.

## Standard Usage
This section contains codeblocks explaining how to use the database. Note that
it's designed so it's easy to copy code when previewed on GitHub, so I suggest
checking it out there:

https://github.com/landoskape/vrAnalysis/tree/main/docs/database.md

### Importing the database and creating a connection
The database module is part of the vrAnalysis package. Import as follows and
create a database object (a one size fits all object for communicating with 
the database).
```
from vrAnalysis import database
vrdb = database.vrDatabase()
```

### Retrieving data from the database
The `getTable` method retrieves data from the database and returns it as a 
pandas dataframe. By default, it ignores "scratched" sessions (those that did 
not pass quality control), and can accept additional kwargs corresponding to 
field names of the database and the desired field value. For example, the
following line will return any session for which suite2p has been performed, 
and ignore any sessions that did not pass quality control.
```
df = vrdb.getTable(suite2p=True)
```

If you simply want to look at the field names of the database, then do this:
```
for fieldName in vrdb.tableData()[0]:
    print(fieldName)
```

Instead of retrieving a dataframe, you can list sessions meeting the desired
criteria. This uses the readable `sessionPrint()` method of `vrSession` 
objects. Here, we list all imaging sessions for mouse ATL020. 
```
vrdb.printSessions(mouseName="ATL020", imaging=True, ignoreScratched=False)
```

Alternatively, if you want an iterable list of `vrExperiment` objects 
referenced in the desired dataframe, you can use this:
```
vrdb.iterSessions(imaging=True, vrRegistration=True)
```

### Registering Sessions
You can register sessions from the `database` object, which is good practice
because it automatically updates the SQL database appropriately. To update a
single session, use this command, where the three arguments are the mousename, 
datestring, and sessionid:
```
vrdb.registerSingleSession('ATL022','2023-04-26','701')
```

To determine which sessions need registration, use this line. Note that 
default behavior is to skip sessions that have already experienced an error in
registration. You can add kwargs to filter the table even further. 
```
vrdb.needsRegistration(skipErrors=True)
```

To register all sessions that still require registration, you can use this 
method. There is a kw argument called `maxData` that restricts how much 
oneData can be produced each time this is run -- (registration saves oneData 
which can be a lot of memory). Additionally, you can specify `userOpts` which
are passed to the `vrRegistration` object that determine how to register each 
session.
```
def registerSessions(self, maxData=30e9, skipErrors=True, **userOpts):
```

If an error was encountered during registration, it is saved in the SQL 
database. To find out what errors have been encountered, use the following 
line. Note that default behavior is to avoid scratched sessions, and usually I
scratch sessions that encountered insurmountable registration errors...
```
vrdb.printRegistrationErrors(ignoreScratched=False)
```

### Managing suite2p information
To determine which sessions require suite2p to be performed (i.e. they are a
session that is not scratched, that has imaging data but has not been 
processed through suite2p), use this: 
```
df = vrdb.needsS2P()
```

If you want to look at sessions in which suite2p _has_ been performed, but has
_not yet_ been quality controlled, use this:
```
df = vrdb.needsS2P(needsQC=True)
```

In addition to both of the above methods, you can also use 
`printRequiresSuite2P`, which prints a list of sessions following the same 
description as above. It also has the `needsQC` switch. Additionally, default 
behavior is `printTargets=True` (a kw argument), which prints the suite2p 
target directories in addition to the session name. 
```
vrdb.printRequiresSuite2P()
```

If suite2p has been run elsewhere or QC'd, you can update the database's 
record of exactly when suite2p files were last updated. (This is useful to 
know when QC was done, and also useful to know if you should re-register any
sessions -- which by the way, I haven't coded yet!). 
```
vrdb.updateSuite2pDateTime()
```

Lastly, if for some reason the database's record of which sessions contain 
processed suite2p data is inaccurate, you can use this method to correct it.
Note that it will only update the database if you set 
`withDatabaseUpdate=True` and will return a boolean telling you if inaccurate 
records were found if you set `returnCheck=True`.
```
validDatabase = vrdb.checkS2P(withDatabaseUpdate=True, returnCheck=True)
```





