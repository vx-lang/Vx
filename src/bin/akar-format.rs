use std::env;
use std::fs;
use std::path::Path;

use akarc::formatter::format_file;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Usage: akar-format <file1.ak> <file2.ak> ...");
        std::process::exit(1);
    }
    
    for file_path in &args[1..] {
        let path = Path::new(file_path);
        if !path.exists() {
            eprintln!("Error: File not found: {}", file_path);
            continue;
        }
        
        match fs::read_to_string(path) {
            Ok(content) => {
                let formatted = format_file(&content);
                if formatted != content {
                    if let Err(e) = fs::write(path, formatted) {
                        eprintln!("Error writing to {}: {}", file_path, e);
                    } else {
                        println!("Formatted {}", file_path);
                    }
                } else {
                    println!("Unchanged {}", file_path);
                }
            }
            Err(e) => {
                eprintln!("Error reading {}: {}", file_path, e);
            }
        }
    }
}
