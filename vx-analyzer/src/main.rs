use lsp_server::{Connection, Message, Request, RequestId, Response};
use lsp_types::{
    Diagnostic, DiagnosticSeverity, DidChangeTextDocumentParams, DidOpenTextDocumentParams, Hover,
    HoverContents, HoverProviderCapability, Position, PublishDiagnosticsParams, Range,
    ServerCapabilities, TextDocumentSyncCapability, TextDocumentSyncKind, Url,
};
use serde_json::Value;
use std::error::Error;
use vxc::ide::{AnalysisHost, IdeDiagnostic};

fn main() -> Result<(), Box<dyn Error + Sync + Send>> {
    let mut log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/Users/adityak/go/Vx/vx-analyzer/analyzer.log")
        .unwrap();
    use std::io::Write;
    writeln!(log_file, "--- vx-analyzer started (lsp-server) ---").unwrap();

    let (connection, io_threads) = Connection::stdio();

    let server_capabilities = serde_json::to_value(&ServerCapabilities {
        text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL)),
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        definition_provider: Some(lsp_types::OneOf::Left(true)),
        references_provider: Some(lsp_types::OneOf::Left(true)),
        document_formatting_provider: Some(lsp_types::OneOf::Left(true)),
        ..Default::default()
    })
    .unwrap();
    let _initialization_params = connection.initialize(server_capabilities)?;

    writeln!(log_file, "Initialized").unwrap();

    main_loop(connection)?;
    io_threads.join()?;

    writeln!(log_file, "Shut down").unwrap();
    Ok(())
}

fn main_loop(connection: Connection) -> Result<(), Box<dyn Error + Sync + Send>> {
    let mut log_file = std::fs::OpenOptions::new()
        .append(true)
        .open("/Users/adityak/go/Vx/vx-analyzer/analyzer.log")
        .unwrap();
    use std::io::Write;

    let mut host = AnalysisHost::new();

    for msg in &connection.receiver {
        writeln!(log_file, "Received msg: {:?}", msg).unwrap();
        match msg {
            Message::Request(req) => {
                if connection.handle_shutdown(&req)? {
                    return Ok(());
                }
                writeln!(log_file, "got request: {:?}", req).unwrap();
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
                        writeln!(log_file, "sending hover response: {:?}", resp).unwrap();
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
                    _ => {}
                }
            }
            Message::Response(_resp) => {
                writeln!(log_file, "got response: {:?}", _resp).unwrap();
            }
            Message::Notification(not) => {
                writeln!(log_file, "got notification: {:?}", not).unwrap();
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
