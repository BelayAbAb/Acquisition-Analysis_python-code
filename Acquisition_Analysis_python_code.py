
import os

# Define the folder structure
folder_structure = {
    '.vscode': {
        'settings.json': ''
    },
    '.github': {
        'workflows': {
            'unittests.yml': ''
        }
    },
    '.gitignore': '',
    'requirements.txt': '',
    'README.md': '',
    'src': {},
    'notebooks': {
        '__init__.py': '',
        'README.md': ''
    },
    'tests': {
        '__init__.py': ''
    },
    'scripts': {
        '__init__.py': '',
        'README.md': ''
    }
}

def create_structure(base_path, structure):
    for name, contents in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(contents, dict):
            # Create directory
            os.makedirs(path, exist_ok=True)
            # Recursively create subdirectories and files
            create_structure(path, contents)
        else:
            # Create file with optional content
            with open(path, 'w') as f:
                f.write(contents)

# Create the folder structure starting from the current directory
create_structure('.', folder_structure)

