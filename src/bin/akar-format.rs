use std::env;
use std::fs;
use std::path::Path;

use akarc::formatter::format_file;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: akar-format [--indent <spaces>] <file1.ak> <file2.ak> ...");
        std::process::exit(1);
    }

    let mut indent_spaces = 2;
    let mut file_paths = Vec::new();

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--indent" && i + 1 < args.len() {
            if let Ok(spaces) = args[i + 1].parse::<usize>() {
                indent_spaces = spaces;
                i += 2;
                continue;
            } else {
                eprintln!("Error: Invalid value for --indent. Must be a number.");
                std::process::exit(1);
            }
        }
        file_paths.push(&args[i]);
        i += 1;
    }

    if file_paths.is_empty() {
        eprintln!("Error: No files provided to format.");
        std::process::exit(1);
    }

    for file_path in file_paths {
        let path = Path::new(file_path);
        if !path.exists() {
            eprintln!("Error: File not found: {}", file_path);
            continue;
        }

        match fs::read_to_string(path) {
            Ok(content) => {
                let formatted = format_file(&content, indent_spaces);
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
