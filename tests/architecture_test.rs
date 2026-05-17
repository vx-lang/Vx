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

        fn process_config<T>(val: T, cfg: &Config, count: u32, limit: u64, ratio: f32) -> &Config {
            return cfg;
        }

        fn run_module_a() -> Config {
            let c = Config { id: 1i32, active: true };
            let count = 100u32;
            let limit = 5000u64;
            let ratio = 1.5f32;
            let _ref = process_config(10i32, &c, count, limit, ratio);
            return c;
        }
        "#,
    )
    .unwrap();

    // File 2: Function with 10+ parameters to trigger Slow Path (UnboundedFunctionMetadata)
    // Also tests ownership references (&mut)
    let file2_path = dir.as_path().join("module_b.vx");
    fs::write(
        &file2_path,
        r#"
        fn compute_heavy(
            p1: &Tensor, p2: &Tensor, p3: &Tensor, p4: &Tensor, 
            p5: &Tensor, p6: &Tensor, p7: &Tensor, p8: &Tensor, 
            p9: &Tensor, p10: &Tensor, p11: &mut Tensor
        ) -> f32 {
            return 1.0f32;
        }

        fn run_module_b(t: Tensor) -> Tensor {
            let mut result = t;
            compute_heavy(&result, &result, &result, &result, &result, &result, &result, &result, &result, &result, &mut result);
            return result;
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
