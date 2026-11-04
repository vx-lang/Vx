mod integration_test {
    // The whole module is macOS-only: it drives CoreML through a Python
    // interpreter that only exists there. Gating the test alone leaves its
    // helpers behind as dead code everywhere else.
    #[cfg(target_os = "macos")]
    mod ane_device_test;
    mod architecture_test;
    mod arrival_counts_test;
    mod assert_codegen_test;
    mod basic_integration;
    mod benchmarks_report_metrics;
    mod borrow_test;
    mod build_commit_in_the_version;
    mod codegen_determinism;
    mod collective_test;
    mod compile_test;
    mod compile_time_of_vec_calls;
    mod comptime_twins;
    mod cross_call_capacity_test;
    mod device_image_test;
    mod device_pool_test;
    mod differential_pairs_hold;
    mod emit_obj_produces_objects;
    mod entry_block_allocas_test;
    mod flat_codegen_differential;
    mod flat_corpus_sweep;
    mod fleet_dtype_test;
    mod fuzz;
    mod gemm_plan_test;
    mod graph_workload_test;
    mod kernel_launch_test;
    mod library_model_check;
    mod lint_test;
    mod loopback_test;
    mod manifest_test;
    mod memory_algebra_axioms;
    mod memory_algebra_fleet;
    mod metadata_test;
    mod missing_solver_test;
    mod mlir_diagnostic_test;
    mod module_api_test;
    // It builds the Mac dispatcher, which needs CoreML.
    #[cfg(target_os = "macos")]
    mod npu_affine_views_test;
    mod output_does_not_depend_on_the_current_directory;
    mod payload_test;
    mod pipeline_scale_test;
    mod registry_test;
    mod remote_client_test;
    mod remote_kernel_output_test;
    mod remote_region_test;
    mod resolution_test;
    mod runtime_test_build;
    mod shipped_programs_compile;
    mod solver_policy_test;
    mod traffic_test;
    mod transport_test;
    // It links with `cc -shared` and loads `.so` files.
    #[cfg(target_os = "linux")]
    mod two_kernel_libraries_in_one_process;
    mod vx_api_diff;
    mod wire_test;
    mod working_set_peak_test;
}
