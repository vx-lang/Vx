use melior::ir::operation::OperationLike;
use melior::ir::BlockLike;
use melior::Context;

fn main() {
    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    let di_file = r#"#llvm.di_file<"test.vx" in "">"#;
    let di_cu = format!(
        r#"#llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = {}, producer = "vx", isOptimized = false, emissionKind = Full>"#,
        di_file
    );
    let di_subp = format!(
        r#"#llvm.di_subprogram<id = distinct[1]<>, compileUnit = {}, scope = {}, name = "test_fn", file = {}, subprogramFlags = Definition, type = #llvm.di_subroutine_type<>>"#,
        di_cu, di_file, di_file
    );

    let mlir_str = format!(
        r#"
    #di_file = {}
    #di_cu = {}
    #di_subp = {}
    module {{
        "dummy.op"() : () -> () loc(fused<#di_subp>["test.vx":1:1])
    }}
    "#,
        di_file, di_cu, di_subp
    );

    let module = melior::ir::Module::parse(&context, &mlir_str).unwrap();
    let op = module.body().first_operation().unwrap();
    let loc = op.location();
    println!("Got loc: {}", loc);
}
