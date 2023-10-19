modules = [
    'fileManagement',
    'helpers',
    'functions',
    'session',
    'database',
    'tracking',
    'analysis',
]

for pkg in modules:
    exec('from . import ' + pkg)
