import re
import os

pattern = re.compile(r'\bcrate::(?:[a-zA-Z0-9_]+::)+[a-zA-Z0-9_]+\b')

def process_file(path):
    with open(path, 'r') as f:
        content = f.read()

    matches = set(pattern.findall(content))
    if not matches:
        return

    imports_to_add = set()
    replacements = {}
    
    for match in matches:
        parts = match.split('::')
        
        import_parts = []
        for i, part in enumerate(parts):
            import_parts.append(part)
            if part[0].isupper() and part != 'crate':
                break
        
        import_stmt = '::'.join(import_parts)
        if import_stmt == match and parts[-1].islower() and len(parts) > 2:
             replacement = parts[-1]
        else:
             replacement = '::'.join(parts[len(import_parts)-1:])
             
        imports_to_add.add(f"use {import_stmt};")
        replacements[match] = replacement

    new_content = content
    for match in sorted(replacements.keys(), key=len, reverse=True):
        new_content = new_content.replace(match, replacements[match])

    lines = new_content.split('\n')
    use_lines = []
    
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('//') or line.startswith('#![') or line.strip() == '':
            insert_idx = i + 1
        elif line.startswith('use '):
            insert_idx = i + 1
        else:
            break
            
    final_imports = []
    for imp in sorted(imports_to_add):
        if imp not in content:
            final_imports.append(imp)
            
    if final_imports:
        lines = lines[:insert_idx] + final_imports + lines[insert_idx:]
        
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Refactored {path}")

for root, dirs, files in os.walk('src'):
    for file in files:
        if file.endswith('.rs'):
            process_file(os.path.join(root, file))
