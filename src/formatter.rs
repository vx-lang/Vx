use crate::lexer::{Lexer, TokenType};

pub fn format_file(content: &str, indent_spaces: usize) -> String {
    let mut lexer = Lexer::new_with_comments(content);
    let mut formatted = String::new();
    let mut indent_level: isize = 0;
    let indent_str = " ".repeat(indent_spaces);
    
    let mut is_new_line = true;
    
    loop {
        let token = lexer.next_token();
        if token.kind == TokenType::Eof {
            break;
        }
        
        match token.kind {
            TokenType::Whitespace(ws) => {
                if ws.contains('\n') {
                    // Output the newlines and reset the start-of-line flag
                    let newlines = ws.chars().filter(|&c| c == '\n').count();
                    for _ in 0..newlines {
                        formatted.push('\n');
                    }
                    is_new_line = true;
                } else {
                    // Only output inline spaces if we aren't at the very start of a line
                    if !is_new_line {
                        formatted.push_str(&ws);
                    }
                }
            }
            TokenType::RightBrace => {
                indent_level -= 1;
                if indent_level < 0 { indent_level = 0; }
                
                if is_new_line {
                    let current_indent = indent_str.repeat(indent_level as usize);
                    formatted.push_str(&current_indent);
                    is_new_line = false;
                }
                
                formatted.push('}');
            }
            TokenType::LeftBrace => {
                if is_new_line {
                    let current_indent = indent_str.repeat(indent_level as usize);
                    formatted.push_str(&current_indent);
                    is_new_line = false;
                }
                
                formatted.push('{');
                indent_level += 1;
            }
            TokenType::Comment(c) => {
                if is_new_line {
                    let current_indent = indent_str.repeat(indent_level as usize);
                    formatted.push_str(&current_indent);
                    is_new_line = false;
                }
                formatted.push_str(&c);
            }
            other => {
                if is_new_line {
                    let current_indent = indent_str.repeat(indent_level as usize);
                    formatted.push_str(&current_indent);
                    is_new_line = false;
                }
                
                formatted.push_str(&other.to_string());
            }
        }
    }
    
    formatted
}
