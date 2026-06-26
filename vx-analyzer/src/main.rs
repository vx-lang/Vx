use std::io::{self, Read, Write};
use vxc::ide::AnalysisHost;

fn main() {
    let mut log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/Users/adityak/go/Vx/vx-analyzer/analyzer.log")
        .unwrap();
    writeln!(log_file, "--- vx-analyzer started ---").unwrap();

    let mut host = AnalysisHost::new();
    let mut stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        // Read headers
        let mut content_length = 0;
        let mut header = String::new();
        let mut byte = [0u8; 1];

        loop {
            if stdin.read_exact(&mut byte).is_err() {
                return;
            }
            let c = byte[0] as char;
            header.push(c);
            if header.ends_with("\r\n\r\n") {
                break;
            }
        }

        // Parse Content-Length
        for line in header.split("\r\n") {
            if line.starts_with("Content-Length: ") {
                if let Ok(len) = line[16..].trim().parse::<usize>() {
                    content_length = len;
                }
            }
        }

        if content_length == 0 {
            continue;
        }

        let mut body = vec![0u8; content_length];
        if stdin.read_exact(&mut body).is_err() {
            return;
        }

        let body_str = String::from_utf8_lossy(&body).to_string();
        writeln!(log_file, "Received payload: {}", body_str).unwrap();

        // Hacky JSON extraction
        let method = extract_string(&body_str, "method");
        let id = extract_number(&body_str, "id");

        if let Some(m) = method {
            if m == "initialize" {
                if let Some(req_id) = id {
                    let response = format!(
                        "{{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{{\"capabilities\":{{\"textDocumentSync\":1,\"hoverProvider\":true,\"definitionProvider\":true,\"referencesProvider\":true}}}}}}",
                        req_id
                    );
                    send_message(&mut stdout, &response);
                }
            } else if m == "initialized" {
                // Log initialized
                log_message(&mut stdout, "vx-analyzer custom initialized!");
            } else if m == "textDocument/didOpen" {
                if let Some(uri) = extract_nested_string(&body_str, "textDocument", "uri") {
                    if let Some(text) = extract_nested_string(&body_str, "textDocument", "text") {
                        let text = unescape_json(&text);
                        host.apply_change(uri.clone(), text.clone());
                        let analysis = host.snapshot();
                        let diagnostics = analysis.diagnostics(&uri);
                        send_diagnostics(&mut stdout, &uri, diagnostics);
                    }
                }
            } else if m == "textDocument/didChange" {
                if let Some(uri) = extract_nested_string(&body_str, "textDocument", "uri") {
                    if let Some(text) = extract_nested_string(&body_str, "contentChanges", "text") {
                        let text = unescape_json(&text);
                        host.apply_change(uri.clone(), text.clone());
                        let analysis = host.snapshot();
                        let diagnostics = analysis.diagnostics(&uri);
                        send_diagnostics(&mut stdout, &uri, diagnostics);
                    }
                }
            } else if m == "textDocument/definition" {
                if let Some(req_id) = id {
                    // MVP: return empty for now
                    let response =
                        format!("{{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":null}}", req_id);
                    send_message(&mut stdout, &response);
                }
            } else if m == "textDocument/hover" {
                if let Some(req_id) = id {
                    if let Some(uri) = extract_nested_string(&body_str, "textDocument", "uri") {
                        let line =
                            extract_nested_number(&body_str, "position", "line").unwrap_or(0);
                        let character =
                            extract_nested_number(&body_str, "position", "character").unwrap_or(0);

                        let analysis = host.snapshot();
                        if let Some(hover) = analysis.hover(&uri, line, character) {
                            let response = format!(
                                "{{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{{\"contents\":{{\"kind\":\"markdown\",\"value\":{}}}}},\"range\":{{\"start\":{{\"line\":{},\"character\":{}}},\"end\":{{\"line\":{},\"character\":{}}}}}}}}}",
                                req_id,
                                escape_json(&hover.value),
                                hover.line_start,
                                hover.col_start,
                                hover.line_end,
                                hover.col_end
                            );
                            send_message(&mut stdout, &response);
                        } else {
                            let response = format!(
                                "{{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":null}}",
                                req_id
                            );
                            send_message(&mut stdout, &response);
                        }
                    }
                }
            } else if m == "textDocument/references" {
                if let Some(req_id) = id {
                    let response =
                        format!("{{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":null}}", req_id);
                    send_message(&mut stdout, &response);
                }
            }
        }
    }
}

fn extract_string(json: &str, key: &str) -> Option<String> {
    let key_str = format!("\"{}\"", key);
    if let Some(idx) = json.find(&key_str) {
        let mut start = idx + key_str.len();
        let bytes = json.as_bytes();

        // Skip spaces and colons
        while start < bytes.len() && (bytes[start] == b' ' || bytes[start] == b':') {
            start += 1;
        }

        if start < bytes.len() && bytes[start] == b'"' {
            start += 1;
            let mut end = start;
            let mut escaped = false;
            while end < bytes.len() {
                if bytes[end] == b'\\' && !escaped {
                    escaped = true;
                } else if bytes[end] == b'"' && !escaped {
                    break;
                } else {
                    escaped = false;
                }
                end += 1;
            }
            return Some(json[start..end].to_string());
        }
    }
    None
}

fn extract_nested_string(json: &str, obj: &str, key: &str) -> Option<String> {
    let key_str = format!("\"{}\"", obj);
    if let Some(idx) = json.find(&key_str) {
        let start = idx + key_str.len();
        extract_string(&json[start..], key)
    } else {
        None
    }
}

fn extract_number(json: &str, key: &str) -> Option<i64> {
    let key_str = format!("\"{}\"", key);
    if let Some(idx) = json.find(&key_str) {
        let mut start = idx + key_str.len();
        let bytes = json.as_bytes();

        while start < bytes.len() && (bytes[start] == b' ' || bytes[start] == b':') {
            start += 1;
        }

        let mut end = start;
        while end < json.len() && json.chars().nth(end).unwrap().is_ascii_digit() {
            end += 1;
        }
        if start < end {
            return json[start..end].parse::<i64>().ok();
        }
    }
    None
}

fn unescape_json(s: &str) -> String {
    let mut res = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(next) = chars.next() {
                match next {
                    'n' => res.push('\n'),
                    'r' => res.push('\r'),
                    't' => res.push('\t'),
                    '\\' => res.push('\\'),
                    '"' => res.push('"'),
                    _ => {
                        res.push('\\');
                        res.push(next);
                    }
                }
            }
        } else {
            res.push(c);
        }
    }
    res
}

fn escape_json(s: &str) -> String {
    let mut res = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\n' => res.push_str("\\n"),
            '\r' => res.push_str("\\r"),
            '\t' => res.push_str("\\t"),
            '\\' => res.push_str("\\\\"),
            '"' => res.push_str("\\\""),
            _ => res.push(c),
        }
    }
    res
}

fn send_message(stdout: &mut std::io::Stdout, msg: &str) {
    write!(stdout, "Content-Length: {}\r\n\r\n{}", msg.len(), msg).unwrap();
    stdout.flush().unwrap();
}

fn log_message(stdout: &mut std::io::Stdout, msg: &str) {
    let escaped = msg.replace("\"", "\\\"");
    let response = format!(
        "{{\"jsonrpc\":\"2.0\",\"method\":\"window/logMessage\",\"params\":{{\"type\":4,\"message\":\"{}\"}}}}",
        escaped
    );
    send_message(stdout, &response);
}

fn extract_nested_number(json: &str, obj: &str, key: &str) -> Option<usize> {
    let key_str = format!("\"{}\"", obj);
    if let Some(idx) = json.find(&key_str) {
        let start = idx + key_str.len();
        extract_number(&json[start..], key).map(|n| n as usize)
    } else {
        None
    }
}

fn send_diagnostics(
    stdout: &mut std::io::Stdout,
    uri: &str,
    diagnostics: Vec<vxc::ide::IdeDiagnostic>,
) {
    let mut diags_json = String::new();
    diags_json.push('[');
    for (i, d) in diagnostics.iter().enumerate() {
        if i > 0 {
            diags_json.push(',');
        }
        let escaped_msg = d.message.replace("\"", "\\\"").replace("\n", "\\n");
        diags_json.push_str(&format!(
            "{{\"range\":{{\"start\":{{\"line\":{},\"character\":{}}},\"end\":{{\"line\":{},\"character\":{}}}}},\"severity\":1,\"message\":\"{}\"}}",
            d.line_start, d.col_start, d.line_end, d.col_end, escaped_msg
        ));
    }
    diags_json.push(']');

    let response = format!(
        "{{\"jsonrpc\":\"2.0\",\"method\":\"textDocument/publishDiagnostics\",\"params\":{{\"uri\":\"{}\",\"diagnostics\":{}}}}}",
        uri, diags_json
    );
    let mut log_file = std::fs::OpenOptions::new()
        .append(true)
        .open("/Users/adityak/go/Vx/vx-analyzer/analyzer.log")
        .unwrap();
    writeln!(log_file, "Sending diagnostics: {}", response).unwrap();
    send_message(stdout, &response);
}
