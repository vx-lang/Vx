import os

path = 'src/ast/macro_expand.rs'
with open(path, 'r') as f:
    content = f.read()

new_content = content.replace('crate::lexer::TokenType::', 'TokenType::')
new_content = new_content.replace('crate::lexer::TokenType', 'TokenType')
new_content = new_content.replace('crate::ast::Delimiter::', 'Delimiter::')
new_content = new_content.replace('crate::ast::Delimiter', 'Delimiter')
new_content = new_content.replace('crate::ast::Span::', 'Span::')
new_content = new_content.replace('crate::ast::Span', 'Span')

# add use
lines = new_content.split('\n')
imports = ["use crate::lexer::TokenType;", "use crate::ast::{Delimiter, Span};"]

for i, line in enumerate(lines):
    if line.startswith('use '):
        lines = lines[:i] + imports + lines[i:]
        break

with open(path, 'w') as f:
    f.write('\n'.join(lines))
