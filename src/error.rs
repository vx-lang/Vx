pub fn format_compiler_error(
    source: &str,
    line: usize,
    col: usize,
    len: usize,
    msg: &str,
) -> String {
    let mut out = format!("Error at {}:{}: {}\n", line, col, msg);
    let lines: Vec<&str> = source.lines().collect();

    if line > 0 && line <= lines.len() {
        let src_line = lines[line - 1];
        out.push_str(src_line);
        out.push('\n');

        let mut pointer = String::new();
        for (i, c) in src_line.chars().enumerate() {
            if i < col - 1 {
                if c == '\t' {
                    pointer.push('\t');
                } else {
                    pointer.push(' ');
                }
            } else if i == col - 1 {
                pointer.push('^');
            } else if i < col - 1 + len {
                pointer.push('~');
            } else {
                break;
            }
        }
        out.push_str(&pointer);
        out.push('\n');
    }
    out
}
