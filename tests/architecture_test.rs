use std::fs;
use std::path::PathBuf;

#[test]
fn test_pipeline_architecture_hooks() {
    let dir = PathBuf::from("tests/modules/architecture_test");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("Failed to create test dir");

    // File 1: Generic functions and structs by value
    let file1_path = dir.join("module_a.vx");
    fs::write(
        &file1_path,
        r#"
        struct Config {
            id: i32,
            active: bool,
        }

        fn process_config<T>(val: T, cfg: Config) -> Config {
            return cfg;
        }

        fn run_module_a() -> Config {
            let c = Config { id: 1, active: true };
            return process_config(10, c);
        }
        "#,
    )
    .unwrap();

    // File 2: Function with 10+ parameters to trigger Slow Path (UnboundedFunctionMetadata)
    // Also tests ownership references (&mut)
    let file2_path = dir.path().join("module_b.vx");
    fs::write(
        &file2_path,
        r#"
        fn compute_heavy(
            p1: Tensor, p2: Tensor, p3: Tensor, p4: Tensor, 
            p5: Tensor, p6: Tensor, p7: Tensor, p8: Tensor, 
            p9: Tensor, p10: Tensor, p11: &mut Tensor
        ) -> Tensor {
            return p1;
        }

        fn run_module_b(t: Tensor) -> Tensor {
            let mut result = t;
            return compute_heavy(t, t, t, t, t, t, t, t, t, t, &mut result);
        }
        "#,
    )
    .unwrap();

    // Collect paths
    let paths = vec![
        file1_path.to_string_lossy().to_string(),
        file2_path.to_string_lossy().to_string(),
    ];

    // Execute pipeline. This will run through the Verification Engine hooks.
    // We expect it to succeed, which means all invariants (Phase 1-8) held true.
    let result = vxc::pipeline::compile_pipeline(&paths);
    assert!(result.is_ok(), "Pipeline failed: {:?}", result.err());
    
    let _ = fs::remove_dir_all(&dir);
}
