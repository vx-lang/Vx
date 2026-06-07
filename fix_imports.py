import os
import re

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    new_content = re.sub(r'^use (ast|parser|sema|registry)(::.*|;)', r'use crate::\1\2', content, flags=re.MULTILINE)
    
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Fixed {filepath}")

if __name__ == "__main__":
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith(".rs"):
                process_file(os.path.join(root, file))
