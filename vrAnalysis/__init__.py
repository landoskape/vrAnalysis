modules = [
    'analysis',
    'database',
    'fileManagement',
    'functions',
    'helpers',
    'session',
]

for pkg in modules:
    exec('from . import ' + pkg)
