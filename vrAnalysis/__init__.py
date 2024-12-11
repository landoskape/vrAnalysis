modules = [
    "fileManagement",
    "helpers",
    "functions",
    "session",
    "registration",
    "uiDatabase",
    "database",
    "tracking",
    "analysis",
    "external",
]

for pkg in modules:
   exec('from . import ' + pkg)
