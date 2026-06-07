import os

path = 'src/pipeline.rs'
with open(path, 'r') as f:
    content = f.read()

new_content = content
new_content = new_content.replace('crate::parallel_architecture_verifier::verify_arch::verify_phase_1_parse', 'verify_phase_1_parse')
new_content = new_content.replace('crate::parallel_architecture_verifier::verify_arch::verify_phase_2_registry', 'verify_phase_2_registry')
new_content = new_content.replace('crate::parallel_architecture_verifier::verify_arch::verify_phase_3_isolation', 'verify_phase_3_isolation')
new_content = new_content.replace('crate::parallel_architecture_verifier::verify_arch::verify_phase_4_deduplication', 'verify_phase_4_deduplication')
new_content = new_content.replace('crate::parallel_architecture_verifier::verify_arch::verify_phase_5_epoch_advance', 'verify_phase_5_epoch_advance')
new_content = new_content.replace('crate::parallel_architecture_verifier::verify_arch::verify_phase_6_simd_patch', 'verify_phase_6_simd_patch')
new_content = new_content.replace('crate::parallel_architecture_verifier::verify_arch::verify_phase_7_routing', 'verify_phase_7_routing')
new_content = new_content.replace('crate::parallel_architecture_verifier::verify_arch::verify_phase_8_serialization', 'verify_phase_8_serialization')
new_content = new_content.replace('crate::diagnostic::DiagnosticLevel::Error', 'DiagnosticLevel::Error')
new_content = new_content.replace('crate::diagnostic::DiagnosticLevel::Warning', 'DiagnosticLevel::Warning')
new_content = new_content.replace('crate::lexer::Lexer::new', 'Lexer::new')
new_content = new_content.replace('crate::parser::Parser::new', 'Parser::new')
new_content = new_content.replace('crate::ast::MacroExpander::new', 'MacroExpander::new')
new_content = new_content.replace('crate::metadata::VxMetadata::save_to_file', 'VxMetadata::save_to_file')
new_content = new_content.replace('crate::session::GlobalSession::new', 'GlobalSession::new')
new_content = new_content.replace('crate::session::LocalWorkerState::new', 'LocalWorkerState::new')
new_content = new_content.replace('crate::sema::GlobalAstEnv::build', 'GlobalAstEnv::build')
new_content = new_content.replace('crate::sema::TypeChecker::new', 'TypeChecker::new')
new_content = new_content.replace('crate::diagnostic::CompilerDiagnostic', 'CompilerDiagnostic')

# add use
lines = new_content.split('\n')
imports = [
    "use crate::parallel_architecture_verifier::verify_arch::*;",
    "use crate::diagnostic::{DiagnosticLevel, CompilerDiagnostic};",
    "use crate::lexer::Lexer;",
    "use crate::parser::Parser;",
    "use crate::ast::MacroExpander;",
    "use crate::metadata::VxMetadata;",
    "use crate::session::{GlobalSession, LocalWorkerState};",
    "use crate::sema::{GlobalAstEnv, TypeChecker};"
]

for i, line in enumerate(lines):
    if line.startswith('use '):
        lines = lines[:i] + imports + lines[i:]
        break

with open(path, 'w') as f:
    f.write('\n'.join(lines))
