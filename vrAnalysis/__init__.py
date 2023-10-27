modules = [
    'fileManagement',
    'helpers',
    'functions',
    'session',
    'registration',
    'database',
    'tracking',
    'analysis',
]

for pkg in modules:
    exec('from . import ' + pkg)
