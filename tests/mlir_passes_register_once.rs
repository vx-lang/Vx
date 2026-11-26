// MLIR keeps its passes in a global map with no lock, so `init_codegen_globals` registers them
// only once. This test is a separate test binary, so the threads below are the first to register
// passes in the process. Each thread does what a compile on that thread does: register the passes,
// then parse a pipeline. When every thread registered the passes itself, most runs of this test
// crashed with "double free or corruption".

#[test]
fn many_threads_register_the_mlir_passes_at_the_same_time() {
    let threads = 14;
    let start = std::sync::Arc::new(std::sync::Barrier::new(threads));
    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let start = start.clone();
            std::thread::spawn(move || {
                start.wait();
                vxc::codegen::init_codegen_globals();
                let context = melior::Context::new();
                let pass_manager = melior::pass::PassManager::new(&context);
                melior::utility::parse_pass_pipeline(
                    pass_manager.as_operation_pass_manager(),
                    "builtin.module(symbol-dce,convert-scf-to-cf,vx-to-llvm)",
                )
                .expect("the pipeline names registered passes");
            })
        })
        .collect();
    for handle in handles {
        handle
            .join()
            .expect("a thread that registers passes panicked");
    }
}
