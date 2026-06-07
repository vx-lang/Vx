import os
import re

pattern = re.compile(r'\bcrate::(ast|parser|registry|sema|codegen)::')

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Find what will be replaced to add imports
    matches = pattern.findall(content)
    if not matches:
        return False
        
    # Replace crate::X:: with X::
    new_content = pattern.sub(r'\1::', content)
    
    if new_content != content:
        # Collect needed imports
        needed = set(matches)
        
        # Add 'use crate::X;' if not already present
        lines = new_content.split('\n')
        
        imports_to_add = []
        for mod in needed:
            imp = f"use crate::{mod};"
            if imp not in content:
                imports_to_add.append(imp)
                
        if imports_to_add:
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('//') or line.startswith('#![') or line.strip() == '':
                    insert_idx = i + 1
                elif line.startswith('use '):
                    insert_idx = i + 1
                else:
                    break
            lines = lines[:insert_idx] + imports_to_add + lines[insert_idx:]
            
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        print(f"Refactored {filepath}")
        return True
    return False

if __name__ == "__main__":
    count = 0
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith(".rs"):
                if process_file(os.path.join(root, file)):
                    count += 1
    print(f"Refactored {count} files.")
