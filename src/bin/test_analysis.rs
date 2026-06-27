use vxc::ide::AnalysisHost;
fn main() {
    let mut host = AnalysisHost::new();
    let uri = "file:///Users/adityak/go/Vx/tests/frontend/pass/closure_stack.vx";
    let text = std::fs::read_to_string("/Users/adityak/go/Vx/tests/frontend/pass/closure_stack.vx")
        .unwrap();

    host.apply_change(uri.to_string(), text);
    let analysis = host.snapshot();
    let hover = analysis.hover(uri, 14, 48);
    println!("Hover result 48: {:?}", hover);

    let def = analysis.goto_definition(uri, 14, 48);
    println!("Definition 48: {:?}", def);
}
