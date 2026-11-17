use super::*;
use crate::parser::formatting::{ClauseKind, RecordedKind, SyntaxRecord};
use crate::parser::Parser;

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
        if self.tokens.iter().enumerate().any(|(index, token)| {
            matches!(token.token.kind, TokenType::Unknown(_))
                && !self.is_excluded(&token.bytes)
                && !self.is_opaque(TokenId(index))
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
            self.find_syntax(&significant);
            self.relax_type_closers(&significant);
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

    /// Parse the file with the compiler's parser and build the syntax views from the token
    /// ranges that the parser records. Excluded text is left out, as `coverage_tokens` says.
    fn find_syntax(&mut self, significant: &[usize]) {
        let (ids, tokens) = self.coverage_tokens(significant);
        let mut records = match Parser::new(&tokens, self.text).formatting_file() {
            Ok(records) => records,
            Err(_) => {
                if Parser::new(&tokens, self.text)
                    .formatting_statements()
                    .is_err()
                {
                    self.warnings.push(CoverageWarning {
                        bytes: 0..self.text.len(),
                        message: "Unsupported source syntax; preserved the file unchanged".into(),
                    });
                }
                return;
            }
        };
        records.sort_by_key(|record| (record.tokens.start, std::cmp::Reverse(record.tokens.end)));
        // The parser reads `tokens`. Convert its ranges to indices into `self.tokens`.
        let to_source = |range: &Range<usize>| {
            if range.is_empty() {
                ids[range.start]..ids[range.start]
            } else {
                ids[range.start]..ids[range.end - 1] + 1
            }
        };
        let inside = |kind: RecordedKind, outer: &Range<usize>| {
            let outer = outer.clone();
            records.iter().filter(move |record| {
                record.kind == kind
                    && outer.start <= record.tokens.start
                    && record.tokens.end <= outer.end
            })
        };
        for declaration in inside(RecordedKind::FunctionDeclaration, &(0..tokens.len())) {
            let header = inside(RecordedKind::FunctionHeader, &declaration.tokens)
                .next()
                .expect("the parser records the header of each function it records");
            let fn_keyword = (header.tokens.clone())
                .find(|index| matches!(tokens[*index].kind, TokenType::Fn))
                .expect("a function header contains `fn`");
            let raw = ids[fn_keyword];
            if self.is_opaque(TokenId(raw)) || self.is_excluded(&self.tokens[raw].bytes) {
                continue;
            }
            let mut start = ids[header.tokens.start];
            if header.tokens.start > 0
                && matches!(tokens[header.tokens.start - 1].kind, TokenType::Pub)
            {
                start = ids[header.tokens.start - 1];
            }
            let terminator = ids[header.tokens.end];
            let declaration_end = ids[declaration.tokens.end - 1] + 1;
            let declaration_bytes = self.token_bytes(start..declaration_end);
            let context = [
                (RecordedKind::ImplBlock, DeclarationContext::Impl),
                (RecordedKind::TraitBlock, DeclarationContext::Trait),
                (RecordedKind::ExternBlock, DeclarationContext::Extern),
            ]
            .into_iter()
            .find(|(kind, _)| {
                records.iter().any(|block| {
                    block.kind == *kind
                        && block.tokens.start <= declaration.tokens.start
                        && declaration.tokens.end <= block.tokens.end
                })
            })
            .map_or(DeclarationContext::Root, |(_, context)| context);
            if inside(RecordedKind::UnreadableBody, &declaration.tokens)
                .next()
                .is_some()
            {
                self.warnings.push(CoverageWarning {
                    bytes: declaration_bytes.clone(),
                    message:
                        "Unsupported function body; preserved the enclosing declaration unchanged"
                            .into(),
                });
            }
            let parameter_list = inside(RecordedKind::ParameterList, &header.tokens)
                .next()
                .expect("the parser records the parameter list of each function header");
            let generics = inside(RecordedKind::GenericParameters, &header.tokens)
                .find(|generics| generics.tokens.start == fn_keyword + 2)
                .map_or(
                    ids[parameter_list.tokens.start]..ids[parameter_list.tokens.start],
                    |generics| to_source(&generics.tokens),
                );
            let return_type = inside(RecordedKind::ReturnType, &header.tokens)
                .next()
                .map_or(terminator..terminator, |ty| to_source(&ty.tokens));
            let clauses = records
                .iter()
                .filter(|record| header.tokens.start <= record.tokens.start)
                .filter(|record| record.tokens.end <= header.tokens.end)
                .filter_map(|record| match record.kind {
                    RecordedKind::Clause(kind) => Some(HeaderClauseView {
                        kind: match kind {
                            ClauseKind::On => HeaderClauseKind::On,
                            ClauseKind::Requires => HeaderClauseKind::Requires,
                            ClauseKind::Ensures => HeaderClauseKind::Ensures,
                            ClauseKind::Where => HeaderClauseKind::Where,
                        },
                        tokens: to_source(&record.tokens),
                    }),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let ineligible = (matches!(
                context,
                DeclarationContext::Trait | DeclarationContext::Extern
            ) && !clauses.is_empty())
                || (context == DeclarationContext::Extern
                    && !matches!(self.tokens[terminator].token.kind, TokenType::Semicolon));
            if ineligible {
                self.warnings.push(CoverageWarning {
                    bytes: declaration_bytes.clone(),
                    message: "Unsupported declaration context; preserved the enclosing declaration unchanged".into(),
                });
            }
            let parameters = inside(RecordedKind::Parameter, &parameter_list.tokens)
                .map(|parameter| ParameterView {
                    tokens: to_source(&parameter.tokens),
                    ty: TypeView {
                        tokens: inside(RecordedKind::ParameterType, &parameter.tokens)
                            .next()
                            .map(|ty| to_source(&ty.tokens))
                            .expect("the parser records the type of each parameter"),
                    },
                })
                .collect();
            self.functions.push(FunctionDeclView {
                context,
                tokens: start..declaration_end,
                bytes: declaration_bytes,
                header: FunctionHeaderView {
                    tokens: start..terminator,
                    name: TokenId(ids[fn_keyword + 1]),
                    generics,
                    parameter_open: TokenId(ids[parameter_list.tokens.start]),
                    parameter_close: TokenId(ids[parameter_list.tokens.end - 1]),
                    parameters,
                    return_type: TypeView {
                        tokens: return_type,
                    },
                    clauses,
                    terminator: TokenId(terminator),
                },
            });
        }
        self.find_struct_literals(&records, &to_source);
        self.find_matches(&records, &to_source);
        self.find_if_values(&records, &to_source);
    }

    fn find_struct_literals(
        &mut self,
        records: &[SyntaxRecord],
        to_source: &dyn Fn(&Range<usize>) -> Range<usize>,
    ) {
        let fields: Vec<_> = records
            .iter()
            .filter(|record| record.kind == RecordedKind::StructField)
            .map(|record| to_source(&record.tokens))
            .collect();
        for literal in records
            .iter()
            .filter(|record| record.kind == RecordedKind::StructLiteral)
        {
            let tokens = to_source(&literal.tokens);
            // The parser also reads `return { ... }` this way. Only a name starts a struct literal.
            if !matches!(
                self.tokens[tokens.start].token.kind,
                TokenType::Identifier(_)
            ) {
                continue;
            }
            let open = (tokens.clone())
                .find(|index| matches!(self.tokens[*index].token.kind, TokenType::LeftBrace))
                .expect("a struct literal has a `{`");
            let close = tokens.end - 1;
            // The fields of this literal, not those of a literal inside one of its values.
            let mut own: Vec<Range<usize>> = Vec::new();
            for field in fields
                .iter()
                .filter(|field| open < field.start && field.end <= close)
            {
                if own.last().is_none_or(|last| last.end <= field.start) {
                    own.push(field.clone());
                }
            }
            let commas = own
                .iter()
                .map(|field| {
                    self.next_significant(field.end)
                        .filter(|index| matches!(self.tokens[*index].token.kind, TokenType::Comma))
                        .map(TokenId)
                })
                .collect();
            let name_end = (tokens.start..open)
                .rev()
                .find(|index| !trivia(&self.tokens[*index].token.kind))
                .expect("a struct literal starts with its name")
                + 1;
            self.struct_literals.push(StructLiteralView {
                name: tokens.start..name_end,
                tokens,
                open: TokenId(open),
                close: TokenId(close),
                fields: own,
                commas,
            });
        }
    }

    fn find_matches(
        &mut self,
        records: &[SyntaxRecord],
        to_source: &dyn Fn(&Range<usize>) -> Range<usize>,
    ) {
        let arms: Vec<_> = records
            .iter()
            .filter(|record| record.kind == RecordedKind::MatchArm)
            .map(|record| to_source(&record.tokens))
            .collect();
        for record in records
            .iter()
            .filter(|record| record.kind == RecordedKind::Match)
        {
            let tokens = to_source(&record.tokens);
            let close = tokens.end - 1;
            let first_arm = arms
                .iter()
                .find(|arm| tokens.start < arm.start && arm.end <= close);
            // A match with no arms keeps the base formatting.
            let Some(open) = first_arm.and_then(|arm| self.previous_significant(arm.start)) else {
                continue;
            };
            assert!(matches!(self.tokens[open].token.kind, TokenType::LeftBrace));
            // The arms of this match, not those of a match inside one of its arms.
            let mut own: Vec<Range<usize>> = Vec::new();
            for arm in arms
                .iter()
                .filter(|arm| open < arm.start && arm.end <= close)
            {
                if own.last().is_none_or(|last| last.end <= arm.start) {
                    own.push(arm.clone());
                }
            }
            let commas = own
                .iter()
                .map(|arm| {
                    self.next_significant(arm.end)
                        .filter(|index| matches!(self.tokens[*index].token.kind, TokenType::Comma))
                        .map(TokenId)
                })
                .collect();
            let scrutinee_start = self
                .next_significant(tokens.start + 1)
                .expect("a match has a value");
            let scrutinee_end = self
                .previous_significant(open)
                .expect("a match has a value")
                + 1;
            self.matches.push(MatchView {
                keyword: TokenId(tokens.start),
                scrutinee: scrutinee_start..scrutinee_end,
                open: TokenId(open),
                close: TokenId(close),
                arms: own,
                commas,
                tokens,
            });
        }
    }

    fn find_if_values(
        &mut self,
        records: &[SyntaxRecord],
        to_source: &dyn Fn(&Range<usize>) -> Range<usize>,
    ) {
        let branches: Vec<_> = records
            .iter()
            .filter(|record| record.kind == RecordedKind::IfValueBranch)
            .map(|record| to_source(&record.tokens))
            .collect();
        for record in records
            .iter()
            .filter(|record| record.kind == RecordedKind::IfValue)
        {
            let tokens = to_source(&record.tokens);
            // Only an `if` where a value goes: after `=`, `(`, `,`, `:`, `return` or `=>`.
            let is_value = self
                .previous_significant(tokens.start)
                .is_some_and(|before| {
                    matches!(
                        self.tokens[before].token.kind,
                        TokenType::Equals
                            | TokenType::LeftParen
                            | TokenType::Comma
                            | TokenType::Colon
                            | TokenType::Return
                            | TokenType::FatArrow
                    )
                });
            if !is_value {
                continue;
            }
            // The two branches of this `if` are the last outermost branches inside it.
            let mut own: Vec<Range<usize>> = Vec::new();
            for branch in branches
                .iter()
                .filter(|branch| tokens.start < branch.start && branch.end <= tokens.end)
            {
                if own.last().is_none_or(|last| last.end <= branch.start) {
                    own.push(branch.clone());
                }
            }
            let [.., then_branch, else_branch] = own.as_slice() else {
                unreachable!("the parser records two branches for each if value");
            };
            let branch = |range: &Range<usize>| {
                let open = range.start;
                let close = range.end - 1;
                let value_start = self
                    .next_significant(open + 1)
                    .expect("a branch has a value");
                let value_end = self
                    .previous_significant(close)
                    .expect("a branch has a value")
                    + 1;
                (TokenId(open), value_start..value_end, TokenId(close))
            };
            let condition_start = self
                .next_significant(tokens.start + 1)
                .expect("an if has a condition");
            let condition_end = self
                .previous_significant(then_branch.start)
                .expect("an if has a condition")
                + 1;
            let else_keyword = self
                .next_significant(then_branch.end)
                .expect("an if value has `else`");
            self.if_values.push(IfValueView {
                condition: condition_start..condition_end,
                branches: [branch(then_branch), branch(else_branch)],
                else_keyword: TokenId(else_keyword),
                tokens,
            });
        }
    }

    /// The last token before `before` that is not whitespace or a comment.
    fn previous_significant(&self, before: usize) -> Option<usize> {
        (0..before)
            .rev()
            .find(|index| !trivia(&self.tokens[*index].token.kind))
    }

    /// The first token from `from` on that is not whitespace or a comment.
    fn next_significant(&self, from: usize) -> Option<usize> {
        (from..self.tokens.len()).find(|index| !trivia(&self.tokens[*index].token.kind))
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
}
