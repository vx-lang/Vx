use std::fs;
use std::path::Path;
use vxc::sema::TypeChecker;

fn main() {
    let path = Path::new("tests/frontend/fail/illegal_transfer_sram_dram.vx");
    let mut loader = vxc::module_loader::ModuleLoader::new();
    let mut program_arr = loader.load_main(path.to_str().unwrap()).unwrap();
    let ast_idx = program_arr
        .iter()
        .position(|p| p.module_path == path.to_str().unwrap())
        .unwrap();
    let mut program = program_arr.remove(ast_idx);

    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));
    let mut all_programs = program_arr.clone();
    all_programs.push(program.clone());
    let env = vxc::sema::GlobalAstEnv::build(&all_programs);
    let mut worker = vxc::session::LocalWorkerState::new(global_session.clone());
    let mut checker = TypeChecker::new(&env, &mut worker);
    for f in &mut program.functions {
        checker.check_function(f);
    }
    
    println!("Errors: {:#?}", checker.errors);
}
