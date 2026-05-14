pub fn format_file(content: &str) -> String {
    let mut formatted = String::new();
    let mut indent_level: isize = 0;
    let indent_str = "    "; // 4 spaces
    
    for line in content.lines() {
        let trimmed = line.trim();
        
        if trimmed.is_empty() {
            formatted.push('\n');
            continue;
        }
        
        // Count closing braces at the start of the line
        let mut start_close_braces = 0;
        for c in trimmed.chars() {
            if c == '}' {
                start_close_braces += 1;
            } else {
                break;
            }
        }
        
        // Decrease indent for starting braces
        indent_level -= start_close_braces;
        if indent_level < 0 {
            indent_level = 0;
        }
        
        // Apply indent
        let current_indent = indent_str.repeat(indent_level as usize);
        formatted.push_str(&current_indent);
        formatted.push_str(trimmed);
        formatted.push('\n');
        
        // Calculate indent change for the rest of the line
        let mut open_braces = 0;
        let mut close_braces = 0;
        
        let mut in_string = false;
        let mut chars = trimmed.chars().peekable();
        
        while let Some(c) = chars.next() {
            if in_string {
                if c == '"' {
                    in_string = false;
                } else if c == '\\' {
                    chars.next(); // skip escaped char
                }
                continue;
            }
            
            if c == '/' && chars.peek() == Some(&'/') {
                break; // line comment, ignore rest
            }
            
            if c == '"' {
                in_string = true;
            } else if c == '{' {
                open_braces += 1;
            } else if c == '}' {
                close_braces += 1;
            }
        }
        
        let remaining_close_braces = close_braces - start_close_braces;
        
        indent_level += open_braces;
        indent_level -= remaining_close_braces;
        
        if indent_level < 0 {
            indent_level = 0;
        }
    }
    
    formatted
}
