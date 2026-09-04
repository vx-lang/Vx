mod integration_test {
    // The whole module is macOS-only: it drives CoreML through a Python
    // interpreter that only exists there. Gating the test alone leaves its
    // helpers behind as dead code everywhere else.
    #[cfg(target_os = "macos")]
    mod ane_device_test;
    mod architecture_test;
    mod assert_codegen_test;
    mod basic_integration;
    mod borrow_test;
    mod codegen_determinism;
    mod compile_test;
    mod cross_call_capacity_test;
    mod device_image_test;
    mod device_pool_test;
    mod entry_block_allocas_test;
    mod flash_routed_test;
    mod flat_codegen_differential;
    mod flat_corpus_sweep;
    mod fleet_dtype_test;
    mod fuzz;
    mod gemm_plan_test;
    mod graph_workload_test;
    mod kernel_launch_test;
    mod lint_test;
    mod loopback_test;
    mod manifest_test;
    mod memory_algebra_axioms;
    mod memory_algebra_fleet;
    mod metadata_test;
    mod missing_solver_test;
    mod mlir_diagnostic_test;
    mod module_api_test;
    mod pipeline_scale_test;
    mod registry_test;
    mod remote_client_test;
    mod remote_region_test;
    mod resolution_test;
    mod solver_policy_test;
    mod traffic_test;
    mod transport_test;
    mod wire_test;
    mod working_set_peak_test;
}
