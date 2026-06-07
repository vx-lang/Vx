import os

path = 'src/driver.rs'
with open(path, 'r') as f:
    content = f.read()

new_content = content
new_content = new_content.replace('crate::ast::MacroExpander::new', 'MacroExpander::new')
new_content = new_content.replace('crate::ast_printer::AstPrinter::print_program', 'AstPrinter::print_program')
new_content = new_content.replace('crate::diagnostic::DiagnosticLevel::Error', 'DiagnosticLevel::Error')
new_content = new_content.replace('crate::diagnostic::DiagnosticLevel::Warning', 'DiagnosticLevel::Warning')
new_content = new_content.replace('crate::codegen::MeliorGenerator::new', 'MeliorGenerator::new')

# add use
lines = new_content.split('\n')
imports = [
    "use crate::ast::MacroExpander;",
    "use crate::ast_printer::AstPrinter;",
    "use crate::diagnostic::DiagnosticLevel;",
    "use crate::codegen::MeliorGenerator;"
]

for i, line in enumerate(lines):
    if line.startswith('use '):
        lines = lines[:i] + imports + lines[i:]
        break

with open(path, 'w') as f:
    f.write('\n'.join(lines))
