import os

EXCLUDE_DIRS = {"venv", ".venv", "env", "__pycache__", ".git"}

def print_tree(root, prefix=""):
    try:
        items = sorted(os.listdir(root))
    except PermissionError:
        return

    items = [item for item in items if item not in EXCLUDE_DIRS]

    for index, item in enumerate(items):
        path = os.path.join(root, item)
        is_last = index == len(items) - 1

        connector = "└── " if is_last else "├── "
        print(prefix + connector + item)

        if os.path.isdir(path):
            extension = "    " if is_last else "│   "
            print_tree(path, prefix + extension)

if __name__ == "__main__":
    root = os.getcwd()
    print(os.path.basename(root) or root)
    print_tree(root)
