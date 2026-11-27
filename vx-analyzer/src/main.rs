use clap::Parser;
use lsp_server::{Connection, ErrorCode, Message, Request, RequestId, Response};
use lsp_types::{
    Diagnostic, DiagnosticSeverity, DidChangeTextDocumentParams, DidOpenTextDocumentParams, Hover,
    HoverContents, HoverProviderCapability, Position, PublishDiagnosticsParams, Range,
    ServerCapabilities, TextDocumentSyncCapability, TextDocumentSyncKind, Url,
};
use serde_json::Value;
use std::error::Error;
use vxc::ide::{AnalysisHost, IdeDiagnostic};

/// Debug log sink, opt-in via `VX_ANALYZER_LOG=<path>`.
///
/// This used to be an unconditional `OpenOptions::open(<an absolute path>).unwrap()`.
/// `create(true)` makes the file, never its parent directory, so on any machine without
/// that directory the server panicked before reading its first message.
fn log_sink() -> Option<std::fs::File> {
    let path = std::env::var_os("VX_ANALYZER_LOG")?;
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .ok()
}

/// Write a line to the sink if there is one. Never fatal: a language server that cannot
/// write its log must still serve.
macro_rules! log_line {
    ($sink:expr, $($arg:tt)*) => {
        if let Some(f) = $sink.as_mut() {
            use std::io::Write;
            let _ = writeln!(f, $($arg)*);
        }
    };
}

/// Command line for `--version`/`--help`. The protocol arrives on stdin, so no argument
/// selects the transport, and `--stdio` (which editors pass) is accepted and ignored.
#[derive(Parser)]
#[command(name = "vx-analyzer", version, about = "Vx language server")]
struct Cli {
    /// Accepted and ignored; hidden so `--help` does not imply another transport exists.
    #[arg(long, hide = true)]
    stdio: bool,
}

fn main() -> Result<(), Box<dyn Error + Sync + Send>> {
    Cli::parse();

    let mut log_file = log_sink();
    log_line!(log_file, "--- vx-analyzer started (lsp-server) ---");

    let (connection, io_threads) = Connection::stdio();

    let server_capabilities = serde_json::to_value(server_capabilities()).unwrap();
    let _initialization_params = connection.initialize(server_capabilities)?;

    log_line!(log_file, "Initialized");

    main_loop(connection)?;
    io_threads.join()?;

    log_line!(log_file, "Shut down");
    Ok(())
}

/// The requests that `main_loop` answers. Add a capability only when `main_loop` handles it.
fn server_capabilities() -> ServerCapabilities {
    ServerCapabilities {
        text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL)),
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        definition_provider: Some(lsp_types::OneOf::Left(true)),
        document_formatting_provider: Some(lsp_types::OneOf::Left(true)),
        ..Default::default()
    }
}

fn main_loop(connection: Connection) -> Result<(), Box<dyn Error + Sync + Send>> {
    let mut log_file = log_sink();

    let mut host = AnalysisHost::new();

    for msg in &connection.receiver {
        log_line!(log_file, "Received msg: {:?}", msg);
        match msg {
            Message::Request(req) => {
                if connection.handle_shutdown(&req)? {
                    return Ok(());
                }
                log_line!(log_file, "got request: {:?}", req);
                match req.method.as_str() {
                    "textDocument/hover" => {
                        let (id, params) = cast::<lsp_types::request::HoverRequest>(req)?;
                        let analysis = host.snapshot();
                        let uri = params
                            .text_document_position_params
                            .text_document
                            .uri
                            .as_str();
                        let line = params.text_document_position_params.position.line as usize;
                        let character =
                            params.text_document_position_params.position.character as usize;

                        let result = if let Some(hover) = analysis.hover(uri, line, character) {
                            Some(Hover {
                                contents: HoverContents::Markup(lsp_types::MarkupContent {
                                    kind: lsp_types::MarkupKind::Markdown,
                                    value: hover.value,
                                }),
                                range: Some(Range {
                                    start: Position::new(
                                        hover.line_start as u32,
                                        hover.col_start as u32,
                                    ),
                                    end: Position::new(hover.line_end as u32, hover.col_end as u32),
                                }),
                            })
                        } else {
                            None
                        };
                        let result = serde_json::to_value(&result).unwrap();
                        let resp = Response {
                            id,
                            result: Some(result),
                            error: None,
                        };
                        log_line!(log_file, "sending hover response: {:?}", resp);
                        connection.sender.send(Message::Response(resp))?;
                    }
                    "textDocument/definition" => {
                        let (id, params) = cast::<lsp_types::request::GotoDefinition>(req)?;
                        let analysis = host.snapshot();
                        let uri = params
                            .text_document_position_params
                            .text_document
                            .uri
                            .as_str();
                        let line = params.text_document_position_params.position.line as usize;
                        let character =
                            params.text_document_position_params.position.character as usize;

                        let result =
                            if let Some(def_loc) = analysis.goto_definition(uri, line, character) {
                                let loc = lsp_types::Location {
                                    uri: lsp_types::Url::parse(&def_loc.uri).unwrap_or_else(|_| {
                                        params
                                            .text_document_position_params
                                            .text_document
                                            .uri
                                            .clone()
                                    }),
                                    range: Range {
                                        start: Position::new(
                                            def_loc.line_start as u32,
                                            def_loc.col_start as u32,
                                        ),
                                        end: Position::new(
                                            def_loc.line_end as u32,
                                            def_loc.col_end as u32,
                                        ),
                                    },
                                };
                                Some(lsp_types::GotoDefinitionResponse::Scalar(loc))
                            } else {
                                None
                            };
                        let result = serde_json::to_value(&result).unwrap();
                        let resp = Response {
                            id,
                            result: Some(result),
                            error: None,
                        };
                        connection.sender.send(Message::Response(resp))?;
                    }
                    "textDocument/formatting" => {
                        let (id, params) = cast::<lsp_types::request::Formatting>(req)?;
                        let uri = params.text_document.uri.as_str();
                        let analysis = host.snapshot();

                        if let Some(text) = analysis.files.get(uri) {
                            let formatted =
                                vxc::formatter::format_file(text, params.options.tab_size as usize);
                            let lines: Vec<&str> = text.lines().collect();
                            let last_line = lines.len().saturating_sub(1) as u32;
                            let last_col = lines.last().map(|s| s.len() as u32).unwrap_or(0);

                            let edit = lsp_types::TextEdit {
                                range: Range {
                                    start: Position::new(0, 0),
                                    end: Position::new(last_line, last_col),
                                },
                                new_text: formatted,
                            };
                            let result = serde_json::to_value(vec![edit]).unwrap();
                            let resp = Response {
                                id,
                                result: Some(result),
                                error: None,
                            };
                            connection.sender.send(Message::Response(resp))?;
                        } else {
                            let resp = Response {
                                id,
                                result: Some(Value::Null),
                                error: None,
                            };
                            connection.sender.send(Message::Response(resp))?;
                        }
                    }
                    _ => {
                        // The protocol needs an answer to every request.
                        let resp = Response::new_err(
                            req.id,
                            ErrorCode::MethodNotFound as i32,
                            format!("vx-analyzer does not handle {}", req.method),
                        );
                        connection.sender.send(Message::Response(resp))?;
                    }
                }
            }
            Message::Response(_resp) => {
                log_line!(log_file, "got response: {:?}", _resp);
            }
            Message::Notification(not) => {
                log_line!(log_file, "got notification: {:?}", not);
                match not.method.as_str() {
                    "textDocument/didOpen" => {
                        let params: DidOpenTextDocumentParams = serde_json::from_value(not.params)?;
                        let uri = params.text_document.uri.as_str().to_string();
                        let text = params.text_document.text;
                        host.apply_change(uri.clone(), text);

                        let analysis = host.snapshot();
                        let diagnostics = analysis.diagnostics(&uri);
                        send_diagnostics(&connection, params.text_document.uri, diagnostics)?;
                    }
                    "textDocument/didChange" => {
                        let mut params: DidChangeTextDocumentParams =
                            serde_json::from_value(not.params)?;
                        let uri = params.text_document.uri.as_str().to_string();
                        if let Some(change) = params.content_changes.pop() {
                            let text = change.text;
                            host.apply_change(uri.clone(), text);

                            let analysis = host.snapshot();
                            let diagnostics = analysis.diagnostics(&uri);
                            send_diagnostics(&connection, params.text_document.uri, diagnostics)?;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

fn cast<R>(req: Request) -> Result<(RequestId, R::Params), Box<dyn Error + Sync + Send>>
where
    R: lsp_types::request::Request,
    R::Params: serde::de::DeserializeOwned,
{
    req.extract(R::METHOD)
        .map_err(|e| format!("cast failed: {:?}", e).into())
}

fn send_diagnostics(
    connection: &Connection,
    uri: Url,
    diagnostics: Vec<IdeDiagnostic>,
) -> Result<(), Box<dyn Error + Sync + Send>> {
    let diags: Vec<Diagnostic> = diagnostics
        .into_iter()
        .map(|d| Diagnostic {
            range: Range {
                start: Position::new(d.line_start as u32, d.col_start as u32),
                end: Position::new(d.line_end as u32, d.col_end as u32),
            },
            severity: Some(DiagnosticSeverity::ERROR),
            code: None,
            code_description: None,
            source: Some("vxc".to_string()),
            message: d.message,
            related_information: None,
            tags: None,
            data: None,
        })
        .collect();

    let params = PublishDiagnosticsParams {
        uri,
        diagnostics: diags,
        version: None,
    };

    let not = lsp_server::Notification::new("textDocument/publishDiagnostics".to_string(), params);
    connection.sender.send(Message::Notification(not))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn a_request_the_server_does_not_handle_gets_a_method_not_found_error() {
        let (server, client) = Connection::memory();
        let server_thread = std::thread::spawn(move || main_loop(server));

        let request = Request::new(
            RequestId::from(7),
            "textDocument/documentSymbol".to_string(),
            serde_json::json!({ "textDocument": { "uri": "file:///add_one.vx" } }),
        );
        client.sender.send(Message::Request(request)).unwrap();

        // Fail after 10 seconds instead of waiting forever when the server sends no answer.
        let message = client
            .receiver
            .recv_timeout(Duration::from_secs(10))
            .expect("the server did not answer the request");
        let Message::Response(response) = message else {
            panic!("expected a response, got {message:?}");
        };
        assert_eq!(response.id, RequestId::from(7));
        assert_eq!(response.result, None);
        let error = response.error.expect("expected an error response");
        assert_eq!(error.code, ErrorCode::MethodNotFound as i32);

        // Closing the client ends the server loop.
        drop(client);
        server_thread.join().unwrap().unwrap();
    }

    #[test]
    fn the_server_does_not_advertise_find_references() {
        // `main_loop` has no code for `textDocument/references`.
        assert_eq!(server_capabilities().references_provider, None);
    }
}
