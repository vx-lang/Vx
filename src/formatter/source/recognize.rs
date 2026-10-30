use super::*;
use crate::parser::{formatting::ClauseKind, Parser};

fn trivia(kind: &TokenType<'_>) -> bool {
    matches!(
        kind,
        TokenType::Whitespace(_) | TokenType::Comment(_) | TokenType::DocComment(_)
    )
}

impl<'source> SourceFileView<'source> {
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
        if self.tokens.iter().any(|token| {
            matches!(token.token.kind, TokenType::Unknown(_)) && !self.is_excluded(&token.bytes)
        }) {
            self.warnings.push(CoverageWarning {
                bytes: 0..self.text.len(),
                message: "Unrecognized lexer token; preserve the file".into(),
            });
        }
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
                if let Some(next) = significant
                    .get(significant.partition_point(|index| *index < end))
                    .copied()
                {
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
        let input: Vec<_> = significant
            .iter()
            .map(|index| self.tokens[*index].token.clone())
            .collect();
        let contexts = self.function_contexts(significant, &input);
        let (coverage_ids, coverage) = self.coverage_tokens(significant);
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
            let mut parser = Parser::new(&input[parser_start..], self.text);
            let (context, inherited) = &contexts[&position];
            let context = *context;
            parser.formatting_inherit_generics(inherited.clone());
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
            let body_start = coverage_ids.partition_point(|id| *id < terminator);
            let body_end = coverage_ids.partition_point(|id| *id < declaration_end);
            let mut body = Parser::new(&coverage[body_start..body_end], self.text);
            body.formatting_inherit_generics(parser.formatting_generic_names());
            if body.formatting_body().is_err() {
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

    fn function_contexts(
        &self,
        significant: &[usize],
        input: &[Token<'_>],
    ) -> std::collections::BTreeMap<usize, (DeclarationContext, Vec<String>)> {
        let mut result = std::collections::BTreeMap::new();
        let mut stack = vec![(DeclarationContext::Root, Vec::new())];
        let mut pending = None;
        for (position, index) in significant.iter().copied().enumerate() {
            if self.is_opaque(TokenId(index)) {
                continue;
            }
            let kind = match self.tokens[index].token.kind {
                TokenType::Impl => Some(DeclarationContext::Impl),
                TokenType::Trait => Some(DeclarationContext::Trait),
                TokenType::Extern => Some(DeclarationContext::Extern),
                TokenType::Struct | TokenType::Enum | TokenType::Memory | TokenType::Topology => {
                    Some(DeclarationContext::Other)
                }
                _ => None,
            };
            if let Some(kind) = kind {
                pending.get_or_insert((kind, position));
            }
            match self.tokens[index].token.kind {
                TokenType::Fn => {
                    result.insert(position, stack.last().unwrap().clone());
                }
                TokenType::LeftBrace => {
                    let mut scope = stack.last().unwrap().clone();
                    if let Some((kind, start)) = pending.take() {
                        scope.0 = kind;
                        if matches!(kind, DeclarationContext::Impl | DeclarationContext::Trait) {
                            if let Ok(names) = Parser::new(&input[start..position], self.text)
                                .formatting_scope_generics()
                            {
                                scope.1.extend(names);
                            }
                        }
                    }
                    stack.push(scope);
                }
                TokenType::RightBrace => {
                    if stack.len() > 1 {
                        stack.pop();
                    }
                    pending = None;
                }
                TokenType::Semicolon => pending = None,
                _ => {}
            }
        }
        result
    }

    /// Ignore excluded text in grammar probes, but retain delimiters that connect to outside code.
    fn coverage_tokens(&self, significant: &[usize]) -> (Vec<usize>, Vec<Token<'source>>) {
        let mut ids = Vec::new();
        let mut tokens = Vec::new();
        for index in significant.iter().copied() {
            let token = &self.tokens[index];
            let excluded = self.is_excluded(&token.bytes);
            if excluded {
                let connects = self.matching[index]
                    .is_some_and(|partner| !self.is_excluded(&self.tokens[partner].bytes));
                if !connects {
                    continue;
                }
                if matches!(token.token.kind, TokenType::LeftBrace) {
                    let mut block = token.token.clone();
                    block.kind = TokenType::Loop;
                    ids.push(index);
                    tokens.push(block);
                }
            }
            ids.push(index);
            tokens.push(token.token.clone());
        }
        (ids, tokens)
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
        let mut owner = vec![None; self.tokens.len()];
        for (id, range) in types.iter().enumerate() {
            for item in &mut owner[range.clone()] {
                *item = Some(id);
            }
        }
        let mut changes = vec![0i32; self.tokens.len() + 1];
        for (index, token) in self.tokens.iter().enumerate() {
            if owner[index].is_some()
                && matches!(
                    token.token.kind,
                    TokenType::LeftBracket | TokenType::LeftBrace
                )
            {
                if let Some(close) = self.matching[index] {
                    changes[index] += 1;
                    changes[close] -= 1;
                }
            }
        }
        let mut depth = 0;
        let expression: Vec<_> = changes
            .into_iter()
            .map(|change| {
                depth += change;
                depth > 0
            })
            .collect();
        for gap in &mut self.gaps {
            if !matches!(
                (
                    &self.tokens[gap.left.0].token.kind,
                    &self.tokens[gap.right.0].token.kind
                ),
                (TokenType::RightAngle, TokenType::RightAngle)
            ) || expression[gap.left.0]
            {
                continue;
            }
            let Some(id) = owner[gap.left.0] else {
                continue;
            };
            if owner[gap.right.0] != Some(id) {
                continue;
            }
            let next = significant.get(significant.partition_point(|index| *index <= gap.right.0));
            if next.is_none_or(|index| {
                *index >= types[id].end
                    || matches!(
                        self.tokens[*index].token.kind,
                        TokenType::Comma | TokenType::RightAngle | TokenType::RightParen
                    )
            }) {
                gap.constraint = GapConstraint::Flexible;
            }
        }
    }

    fn check_remaining_syntax(&mut self, significant: &[usize]) {
        let (ids, tokens) = self.coverage_tokens(significant);
        let mut ranges: Vec<_> = self
            .functions
            .iter()
            .map(|function| function.tokens.clone())
            .collect();
        ranges.sort_by_key(|range| range.start);
        let mut cursor = 0;
        let input: Vec<_> = ids
            .iter()
            .zip(tokens)
            .filter_map(|(index, token)| {
                while cursor < ranges.len() && ranges[cursor].end <= *index {
                    cursor += 1;
                }
                if ranges
                    .get(cursor)
                    .is_some_and(|range| range.contains(index))
                {
                    None
                } else {
                    Some(token)
                }
            })
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
