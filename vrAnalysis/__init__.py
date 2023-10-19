modules = [
    'fileManagement',
    'helpers',
    'functions',
    'database',
    'session',
    'tracking',
    'analysis',
]

for pkg in modules:
    exec('from . import ' + pkg)
