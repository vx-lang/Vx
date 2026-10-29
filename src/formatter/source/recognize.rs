use super::*;
use crate::parser::{formatting::ClauseKind, Parser};

fn trivia(kind: &TokenType<'_>) -> bool {
    matches!(
        kind,
        TokenType::Whitespace(_) | TokenType::Comment(_) | TokenType::DocComment(_)
    )
}

impl SourceFileView<'_> {
    pub(super) fn recognize(&mut self) {
        self.match_delimiters();
        let significant: Vec<usize> = self
            .tokens
            .iter()
            .enumerate()
            .filter(|(_, token)| !trivia(&token.token.kind))
            .map(|(index, _)| index)
            .collect();
        self.find_macros(&significant);
        self.find_exclusions();
        self.find_gaps(&significant);
        if !self
            .warnings
            .iter()
            .any(|warning| warning.bytes == (0..self.text.len()))
        {
            self.find_functions(&significant);
            self.relax_type_closers(&significant);
            self.check_remaining_syntax(&significant);
        }
    }

    fn match_delimiters(&mut self) {
        let mut stack = Vec::new();
        for (index, token) in self.tokens.iter().enumerate() {
            let (kind, opening) = match token.token.kind {
                TokenType::LeftParen => (0, true),
                TokenType::RightParen => (0, false),
                TokenType::LeftBracket => (1, true),
                TokenType::RightBracket => (1, false),
                TokenType::LeftBrace => (2, true),
                TokenType::RightBrace => (2, false),
                _ => continue,
            };
            if opening {
                stack.push((kind, index));
            } else if let Some((before, start)) = stack.pop() {
                if before == kind {
                    self.matching[start] = Some(index);
                    self.matching[index] = Some(start);
                } else {
                    self.warnings.push(CoverageWarning {
                        bytes: 0..self.text.len(),
                        message: "Unrecognized delimiter structure; preserve the file".into(),
                    });
                }
            } else {
                self.warnings.push(CoverageWarning {
                    bytes: 0..self.text.len(),
                    message: "Unmatched delimiter; preserve the file".into(),
                });
            }
        }
        if !stack.is_empty() {
            self.warnings.push(CoverageWarning {
                bytes: 0..self.text.len(),
                message: "Incomplete delimiter structure; preserve the file".into(),
            });
        }
        if self
            .tokens
            .iter()
            .any(|token| matches!(token.token.kind, TokenType::Unknown(_)))
        {
            self.warnings.push(CoverageWarning {
                bytes: 0..self.text.len(),
                message: "Unrecognized lexer token; preserve the file".into(),
            });
        }
    }

    fn find_macros(&mut self, significant: &[usize]) {
        let mut position = 0;
        while position + 2 < significant.len() {
            let start = significant[position];
            let definition = matches!(self.tokens[start].token.kind, TokenType::MacroRules);
            let invocation = matches!(self.tokens[start].token.kind, TokenType::Identifier(_))
                && matches!(
                    self.tokens[significant[position + 1]].token.kind,
                    TokenType::Bang
                );
            if !definition && !invocation {
                position += 1;
                continue;
            }
            let open = significant[position + 2];
            if !matches!(
                self.tokens[open].token.kind,
                TokenType::LeftParen | TokenType::LeftBracket | TokenType::LeftBrace
            ) || (definition && !matches!(self.tokens[open].token.kind, TokenType::LeftBrace))
            {
                self.warnings.push(CoverageWarning {
                    bytes: 0..self.text.len(),
                    message: "Unsupported macro envelope; preserve the file".into(),
                });
                break;
            }
            if invocation
                && self.tokens[start].bytes.end
                    != self.tokens[significant[position + 1]].bytes.start
            {
                self.warnings.push(CoverageWarning {
                    bytes: 0..self.text.len(),
                    message: "Unrecognized separated macro name; preserve the file".into(),
                });
                break;
            }
            let Some(close) = self.matching[open] else {
                self.warnings.push(CoverageWarning {
                    bytes: 0..self.text.len(),
                    message: "Incomplete macro envelope; preserve the file".into(),
                });
                break;
            };
            let mut end = close + 1;
            let mut trees: Vec<_> = std::iter::once(open..end).collect();
            if !definition {
                if let Some(next) = significant.iter().copied().find(|index| *index >= end) {
                    if matches!(self.tokens[next].token.kind, TokenType::LeftBrace) {
                        if let Some(block_close) = self.matching[next] {
                            trees.push(next..block_close + 1);
                            end = block_close + 1;
                        }
                    }
                }
            } else {
                trees.clear();
                let mut cursor = open + 1;
                while cursor < close {
                    if matches!(
                        self.tokens[cursor].token.kind,
                        TokenType::LeftParen | TokenType::LeftBrace
                    ) {
                        if let Some(tree_close) = self.matching[cursor] {
                            trees.push(cursor..tree_close + 1);
                            cursor = tree_close + 1;
                            continue;
                        }
                    }
                    cursor += 1;
                }
            }
            let token = &self.tokens[start].token;
            self.macros.push(MacroView {
                kind: if definition {
                    MacroKind::Definition
                } else {
                    MacroKind::Invocation
                },
                tokens: start..end,
                bytes: self.token_bytes(start..end),
                token_trees: trees,
                original_line: token.line,
                original_column: token.column,
            });
            while position < significant.len() && significant[position] < end {
                position += 1;
            }
        }
    }

    fn find_exclusions(&mut self) {
        let mut begin = None;
        for (index, token) in self.tokens.iter().enumerate() {
            if self.is_opaque(TokenId(index)) {
                continue;
            }
            let text = match &token.token.kind {
                TokenType::Comment(text) | TokenType::DocComment(text) => text,
                _ => continue,
            };
            if index == 0 && text.contains("//! vx-format: OFF") {
                self.exclusions = std::iter::once(0..self.text.len()).collect();
                return;
            }
            if text.contains("vx-format-begin: OFF") {
                let line_start = self.text[..token.bytes.start]
                    .rfind('\n')
                    .map_or(0, |offset| offset + 1);
                let start = if self.text[line_start..token.bytes.start]
                    .chars()
                    .all(char::is_whitespace)
                {
                    line_start
                } else {
                    token.bytes.start
                };
                begin.get_or_insert(start);
            } else if text.contains("vx-format-begin: ON") || text.contains("vx-format-end: OFF") {
                if let Some(start) = begin.take() {
                    let end = token.bytes.end
                        + usize::from(self.text[token.bytes.end..].starts_with('\n'));
                    self.exclusions.push(start..end);
                }
            }
        }
        if let Some(start) = begin {
            self.exclusions.push(start..self.text.len());
        }
    }

    fn find_gaps(&mut self, significant: &[usize]) {
        for pair in significant.windows(2) {
            let left = &self.tokens[pair[0]];
            let right = &self.tokens[pair[1]];
            let sensitive = matches!(
                (&left.token.kind, &right.token.kind),
                (
                    TokenType::LeftAngle,
                    TokenType::LeftAngle | TokenType::LessEq
                ) | (
                    TokenType::RightAngle,
                    TokenType::RightAngle | TokenType::GreaterEq
                ) | (TokenType::Identifier(_), TokenType::Bang)
            );
            if sensitive {
                self.gaps.push(ConstrainedGap {
                    left: TokenId(pair[0]),
                    right: TokenId(pair[1]),
                    constraint: if left.bytes.end == right.bytes.start {
                        GapConstraint::MustTouch
                    } else {
                        GapConstraint::MustSeparate
                    },
                });
            }
        }
    }

    fn find_functions(&mut self, significant: &[usize]) {
        for (position, raw) in significant.iter().copied().enumerate() {
            if !matches!(self.tokens[raw].token.kind, TokenType::Fn)
                || self.is_opaque(TokenId(raw))
                || self.is_excluded(&self.tokens[raw].bytes)
            {
                continue;
            }
            if !significant.get(position + 1).is_some_and(|next| {
                matches!(self.tokens[*next].token.kind, TokenType::Identifier(_))
            }) {
                continue;
            }
            let parser_start = if position > 0
                && matches!(
                    self.tokens[significant[position - 1]].token.kind,
                    TokenType::Unsafe
                ) {
                position - 1
            } else {
                position
            };
            let ids = &significant[parser_start..];
            let input: Vec<_> = ids
                .iter()
                .map(|index| self.tokens[*index].token.clone())
                .collect();
            let mut parser = Parser::new(&input, self.text);
            let (context, inherited) = self.function_context(significant, position);
            parser.formatting_inherit_generics(inherited);
            let parsed = match parser.formatting_header() {
                Ok(parsed) => parsed,
                Err(_) => {
                    self.warnings.push(CoverageWarning {
                        bytes: 0..self.text.len(),
                        message: "Unsupported complete function header; preserve the file".into(),
                    });
                    continue;
                }
            };
            let terminator = ids[parsed.terminator];
            let declaration_end =
                if matches!(self.tokens[terminator].token.kind, TokenType::LeftBrace) {
                    let Some(close) = self.matching[terminator] else {
                        continue;
                    };
                    close + 1
                } else {
                    terminator + 1
                };
            let mut start = ids[0];
            if parser_start > 0
                && matches!(
                    self.tokens[significant[parser_start - 1]].token.kind,
                    TokenType::Pub | TokenType::Safe
                )
            {
                start = significant[parser_start - 1];
            }
            if parser.formatting_body().is_err() {
                self.warnings.push(CoverageWarning {
                    bytes: self.token_bytes(start..declaration_end),
                    message:
                        "Unsupported function body; preserved the enclosing declaration unchanged"
                            .into(),
                });
            }
            let ineligible = context == DeclarationContext::Other
                || (matches!(
                    context,
                    DeclarationContext::Trait | DeclarationContext::Extern
                ) && (!parsed.clauses.is_empty() || parsed.return_type.is_empty()))
                || (context == DeclarationContext::Trait
                    && self.tokens[start..raw].iter().any(|token| {
                        matches!(
                            token.token.kind,
                            TokenType::Pub | TokenType::Unsafe | TokenType::Safe
                        )
                    }))
                || (context == DeclarationContext::Extern
                    && !matches!(self.tokens[terminator].token.kind, TokenType::Semicolon));
            if ineligible {
                self.warnings.push(CoverageWarning { bytes: self.token_bytes(start..declaration_end), message: "Unsupported declaration context; preserved the enclosing declaration unchanged".into() });
            }
            let convert = |range: Range<usize>| {
                if range.is_empty() {
                    let point = ids[range.start];
                    point..point
                } else {
                    ids[range.start]..ids[range.end - 1] + 1
                }
            };
            self.functions.push(FunctionDeclView {
                context,
                tokens: start..declaration_end,
                bytes: self.token_bytes(start..declaration_end),
                header: FunctionHeaderView {
                    tokens: start..terminator,
                    name: TokenId(ids[parsed.name]),
                    generics: convert(parsed.generics),
                    parameter_open: TokenId(ids[parsed.open]),
                    parameter_close: TokenId(ids[parsed.close]),
                    parameters: parsed
                        .parameters
                        .into_iter()
                        .map(|(tokens, ty)| ParameterView {
                            tokens: convert(tokens),
                            ty: TypeView {
                                tokens: convert(ty),
                            },
                        })
                        .collect(),
                    return_type: TypeView {
                        tokens: convert(parsed.return_type),
                    },
                    clauses: parsed
                        .clauses
                        .into_iter()
                        .map(|(kind, tokens)| HeaderClauseView {
                            kind: match kind {
                                ClauseKind::On => HeaderClauseKind::On,
                                ClauseKind::Requires => HeaderClauseKind::Requires,
                                ClauseKind::Ensures => HeaderClauseKind::Ensures,
                                ClauseKind::Where => HeaderClauseKind::Where,
                            },
                            tokens: convert(tokens),
                        })
                        .collect(),
                    terminator: TokenId(terminator),
                },
            });
        }
    }

    fn function_context(
        &self,
        significant: &[usize],
        position: usize,
    ) -> (DeclarationContext, Vec<String>) {
        let at = significant[position];
        let mut scopes = std::collections::BTreeMap::new();
        for (start, index) in significant[..position].iter().copied().enumerate() {
            let kind = match self.tokens[index].token.kind {
                TokenType::Impl => DeclarationContext::Impl,
                TokenType::Trait => DeclarationContext::Trait,
                TokenType::Extern => DeclarationContext::Extern,
                TokenType::Struct | TokenType::Enum | TokenType::Memory | TokenType::Topology => {
                    DeclarationContext::Other
                }
                _ => continue,
            };
            if self.is_opaque(TokenId(index)) {
                continue;
            }
            let open = significant[start + 1..position]
                .iter()
                .copied()
                .find(|candidate| {
                    matches!(self.tokens[*candidate].token.kind, TokenType::LeftBrace)
                })
                .filter(|candidate| self.matching[*candidate].is_some_and(|close| at < close));
            if let Some(open) = open {
                scopes.entry(open).or_insert((kind, start));
            }
        }
        let mut context = DeclarationContext::Root;
        let mut generics = Vec::new();
        for (_, (kind, start)) in scopes {
            context = kind;
            if !matches!(kind, DeclarationContext::Impl | DeclarationContext::Trait) {
                continue;
            }
            let input: Vec<_> = significant[start..]
                .iter()
                .map(|index| self.tokens[*index].token.clone())
                .collect();
            if let Ok(names) = Parser::new(&input, self.text).formatting_scope_generics() {
                generics.extend(names);
            }
        }
        (context, generics)
    }

    fn relax_type_closers(&mut self, significant: &[usize]) {
        let types: Vec<_> = self
            .functions
            .iter()
            .flat_map(|function| {
                function
                    .header
                    .parameters
                    .iter()
                    .map(|parameter| parameter.ty.tokens.clone())
                    .chain(std::iter::once(function.header.return_type.tokens.clone()))
                    .chain(std::iter::once(function.header.generics.clone()))
            })
            .collect();
        for gap in &mut self.gaps {
            if !matches!(
                (
                    &self.tokens[gap.left.0].token.kind,
                    &self.tokens[gap.right.0].token.kind
                ),
                (TokenType::RightAngle, TokenType::RightAngle)
            ) {
                continue;
            }
            let in_expression = self.tokens.iter().enumerate().any(|(index, token)| {
                matches!(
                    token.token.kind,
                    TokenType::LeftBracket | TokenType::LeftBrace
                ) && index < gap.left.0
                    && self.matching[index].is_some_and(|close| gap.right.0 < close)
                    && types.iter().any(|range| range.contains(&index))
            });
            if in_expression {
                continue;
            }
            let next = significant
                .iter()
                .copied()
                .find(|index| *index > gap.right.0);
            let closes_type = types.iter().any(|range| {
                range.contains(&gap.left.0)
                    && range.contains(&gap.right.0)
                    && next.is_none_or(|index| {
                        index >= range.end
                            || matches!(
                                self.tokens[index].token.kind,
                                TokenType::Comma | TokenType::RightAngle | TokenType::RightParen
                            )
                    })
            });
            if closes_type {
                gap.constraint = GapConstraint::Flexible;
            }
        }
    }

    fn check_remaining_syntax(&mut self, significant: &[usize]) {
        let input: Vec<_> = significant
            .iter()
            .copied()
            .filter(|index| {
                !self
                    .functions
                    .iter()
                    .any(|function| function.tokens.contains(index))
            })
            .map(|index| self.tokens[index].token.clone())
            .collect();
        // AST results are discarded. Source views remain authoritative for every output range.
        if Parser::new(&input, self.text).parse().is_err()
            && Parser::new(&input, self.text)
                .formatting_statements()
                .is_err()
        {
            self.warnings.push(CoverageWarning {
                bytes: 0..self.text.len(),
                message: "Unsupported source syntax; preserved the file unchanged".into(),
            });
        }
    }
}
