use vxc::gid::TypeId;
use vxc::borrow::verify_subtyping_bounds;
use vxc::session::CompilationSession;

#[test]
fn test_fast_path_variance_checks() {
    let session = CompilationSession::new(1);
    // Type A: Variance = 0, Region = 0 ('static)
    let mut type_a = TypeId::new(0, 0, 0, 0);
    type_a.try_set_fast_param(0, 0, 0).unwrap();

    // Type B: Variance = 0, Region = 1 ('a)
    let mut type_b = TypeId::new(0, 0, 0, 0);
    type_b.try_set_fast_param(0, 1, 0).unwrap();

    // 'static (0) can be coerced to 'a (1)
    assert!(verify_subtyping_bounds(&type_a, &type_b, &session));

    // 'a (1) cannot be coerced to 'static (0)
    assert!(!verify_subtyping_bounds(&type_b, &type_a, &session));

    // Exact structural match
    assert!(verify_subtyping_bounds(&type_a, &type_a, &session));
}
