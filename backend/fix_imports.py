import os
from pathlib import Path

backend_dir = Path("/home/kxd/Coding/visionresearch/backend")

for root, _, files in os.walk(backend_dir):
    for file in files:
        if file.endswith(".py") and file != "fix_imports.py":
            filepath = Path(root) / file
            content = filepath.read_text()
            
            # Replace absolute imports starting with backend.
            new_content = content.replace("from backend.", "from ")
            new_content = new_content.replace("import backend.", "import ")
            
            if new_content != content:
                filepath.write_text(new_content)
                print(f"Fixed imports in {filepath.relative_to(backend_dir)}")
