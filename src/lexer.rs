//===- lexer.rs - Vx Compiler ----------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file implements the Lexical Analyzer (Scanner) for the Vx language.
// It processes raw source text and converts it into a stream of discrete tokens
// (keywords, identifiers, literals, symbols), handling whitespace, comments, and
// basic syntax validation.
//
//===----------------------------------------------------------------------===//

#[derive(Debug, PartialEq, Clone, Eq)]
pub enum TokenTypeBase<S, C> {
    // Keywords
    Fn,
    Let,
    Mut,
    For,
    In,
    If,
    Else,
    Loop,
    Break,
    Continue,
    Return,
    Spawn,
    On,
    Transfer,
    Unroll,
    Across,
    Match,
    Struct,
    Unsafe,
    Safe,
    Extern,
    Trait,
    Impl,
    Comptime,
    Import,
    Assert,
    Enum,
    As,
    Grad,
    Vjp,
    Jvp,
    Requires,
    Ensures,
    Invariant,
    MacroRules,

    // Types & Topology
    Topology,
    Memory,
    Ref,
    Verified,
    Pinned,
    HardwareState,

    // Identifiers & Literals
    Identifier(S),
    Number(S),
    StringLiteral(C),

    // Symbols
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    LeftBracket,
    RightBracket,
    LeftAngle,
    RightAngle,
    Colon,
    DoubleColon,
    Semicolon,
    Comma,
    Equals,
    PlusEquals,
    Arrow,
    FatArrow,
    Plus,
    Minus,
    Star,
    Slash,
    Dot,
    DoubleDot,
    Ampersand,
    At,
    Dollar,

    // Logical & Relational
    EqEq,
    NotEq,
    LessEq,
    GreaterEq,
    AndAnd,
    OrOr,
    Bang,
    Pipe,

    // Special
    Eof,
    Unknown(char),
    Comment(S),
    DocComment(S),
    Whitespace(S),
}

pub type TokenType<'a> = TokenTypeBase<&'a str, std::borrow::Cow<'a, str>>;
pub type OwnedTokenType = TokenTypeBase<crate::symbol::Symbol, String>;

impl<S: std::fmt::Display, C: std::fmt::Display> std::fmt::Display for TokenTypeBase<S, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenTypeBase::Fn => write!(f, "fn"),
            TokenTypeBase::Let => write!(f, "let"),
            TokenTypeBase::Mut => write!(f, "mut"),
            TokenTypeBase::For => write!(f, "for"),
            TokenTypeBase::In => write!(f, "in"),
            TokenTypeBase::If => write!(f, "if"),
            TokenTypeBase::Else => write!(f, "else"),
            TokenTypeBase::Loop => write!(f, "loop"),
            TokenTypeBase::Break => write!(f, "break"),
            TokenTypeBase::Continue => write!(f, "continue"),
            TokenTypeBase::Return => write!(f, "return"),
            TokenTypeBase::Spawn => write!(f, "spawn"),
            TokenTypeBase::On => write!(f, "on"),
            TokenTypeBase::Transfer => write!(f, "transfer"),
            TokenTypeBase::Unroll => write!(f, "unroll"),
            TokenTypeBase::Across => write!(f, "across"),
            TokenTypeBase::Match => write!(f, "match"),
            TokenTypeBase::Struct => write!(f, "struct"),
            TokenTypeBase::Unsafe => write!(f, "unsafe"),
            TokenTypeBase::Safe => write!(f, "safe"),
            TokenTypeBase::Extern => write!(f, "extern"),
            TokenTypeBase::Trait => write!(f, "trait"),
            TokenTypeBase::Impl => write!(f, "impl"),
            TokenTypeBase::Comptime => write!(f, "comptime"),
            TokenTypeBase::Import => write!(f, "import"),
            TokenTypeBase::Assert => write!(f, "assert"),
            TokenTypeBase::Enum => write!(f, "enum"),
            TokenTypeBase::As => write!(f, "as"),
            TokenTypeBase::Grad => write!(f, "grad"),
            TokenTypeBase::Vjp => write!(f, "vjp"),
            TokenTypeBase::Jvp => write!(f, "jvp"),
            TokenTypeBase::Requires => write!(f, "requires"),
            TokenTypeBase::Ensures => write!(f, "ensures"),
            TokenTypeBase::Invariant => write!(f, "invariant"),

            TokenTypeBase::Topology => write!(f, "Topology"),
            TokenTypeBase::Memory => write!(f, "Memory"),
            TokenTypeBase::Ref => write!(f, "Ref"),
            TokenTypeBase::Verified => write!(f, "Verified"),
            TokenTypeBase::Pinned => write!(f, "Pinned"),
            TokenTypeBase::HardwareState => write!(f, "HardwareState"),

            TokenTypeBase::MacroRules => write!(f, "macro_rules"),

            TokenTypeBase::Identifier(s) => write!(f, "{}", s),
            TokenTypeBase::Number(s) => write!(f, "{}", s),
            TokenTypeBase::StringLiteral(s) => write!(f, "\"{}\"", s),

            TokenTypeBase::LeftParen => write!(f, "("),
            TokenTypeBase::RightParen => write!(f, ")"),
            TokenTypeBase::LeftBrace => write!(f, "{{"),
            TokenTypeBase::RightBrace => write!(f, "}}"),
            TokenTypeBase::LeftBracket => write!(f, "["),
            TokenTypeBase::RightBracket => write!(f, "]"),
            TokenTypeBase::LeftAngle => write!(f, "<"),
            TokenTypeBase::RightAngle => write!(f, ">"),
            TokenTypeBase::Colon => write!(f, ":"),
            TokenTypeBase::DoubleColon => write!(f, "::"),
            TokenTypeBase::Semicolon => write!(f, ";"),
            TokenTypeBase::Comma => write!(f, ","),
            TokenTypeBase::Equals => write!(f, "="),
            TokenTypeBase::PlusEquals => write!(f, "+="),
            TokenTypeBase::Arrow => write!(f, "->"),
            TokenTypeBase::FatArrow => write!(f, "=>"),
            TokenTypeBase::Plus => write!(f, "+"),
            TokenTypeBase::Minus => write!(f, "-"),
            TokenTypeBase::Star => write!(f, "*"),
            TokenTypeBase::Slash => write!(f, "/"),
            TokenTypeBase::Dot => write!(f, "."),
            TokenTypeBase::DoubleDot => write!(f, ".."),
            TokenTypeBase::Ampersand => write!(f, "&"),
            TokenTypeBase::At => write!(f, "@"),
            TokenTypeBase::Dollar => write!(f, "$"),

            TokenTypeBase::EqEq => write!(f, "=="),
            TokenTypeBase::NotEq => write!(f, "!="),
            TokenTypeBase::LessEq => write!(f, "<="),
            TokenTypeBase::GreaterEq => write!(f, ">="),
            TokenTypeBase::AndAnd => write!(f, "&&"),
            TokenTypeBase::OrOr => write!(f, "||"),
            TokenTypeBase::Bang => write!(f, "!"),
            TokenTypeBase::Pipe => write!(f, "|"),

            TokenTypeBase::Comment(s) => write!(f, "{}", s),
            TokenTypeBase::DocComment(s) => write!(f, "{}", s),
            TokenTypeBase::Whitespace(s) => write!(f, "{}", s),
            TokenTypeBase::Unknown(c) => write!(f, "{}", c),
            TokenTypeBase::Eof => write!(f, ""),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TokenBase<S, C> {
    pub kind: TokenTypeBase<S, C>,
    pub line: usize,
    pub column: usize,
    pub length: usize,
}

pub type Token<'a> = TokenBase<&'a str, std::borrow::Cow<'a, str>>;
pub type OwnedToken = TokenBase<crate::symbol::Symbol, String>;

use once_cell::sync::Lazy;
use rustc_hash::FxHashMap;

static KEYWORDS: Lazy<
    FxHashMap<&'static str, TokenTypeBase<&'static str, std::borrow::Cow<'static, str>>>,
> = Lazy::new(|| {
    let mut m = FxHashMap::default();
    m.insert("fn", TokenTypeBase::Fn);
    m.insert("let", TokenTypeBase::Let);
    m.insert("mut", TokenTypeBase::Mut);
    m.insert("for", TokenTypeBase::For);
    m.insert("in", TokenTypeBase::In);
    m.insert("if", TokenTypeBase::If);
    m.insert("else", TokenTypeBase::Else);
    m.insert("loop", TokenTypeBase::Loop);
    m.insert("break", TokenTypeBase::Break);
    m.insert("continue", TokenTypeBase::Continue);
    m.insert("return", TokenTypeBase::Return);
    m.insert("spawn", TokenTypeBase::Spawn);
    m.insert("on", TokenTypeBase::On);
    m.insert("transfer", TokenTypeBase::Transfer);
    m.insert("unroll", TokenTypeBase::Unroll);
    m.insert("across", TokenTypeBase::Across);
    m.insert("match", TokenTypeBase::Match);
    m.insert("Topology", TokenTypeBase::Topology);
    m.insert("Memory", TokenTypeBase::Memory);
    m.insert("Ref", TokenTypeBase::Ref);
    m.insert("Verified", TokenTypeBase::Verified);
    m.insert("Pinned", TokenTypeBase::Pinned);
    m.insert("HardwareState", TokenTypeBase::HardwareState);
    m.insert("struct", TokenTypeBase::Struct);
    m.insert("unsafe", TokenTypeBase::Unsafe);
    m.insert("safe", TokenTypeBase::Safe);
    m.insert("extern", TokenTypeBase::Extern);
    m.insert("trait", TokenTypeBase::Trait);
    m.insert("impl", TokenTypeBase::Impl);
    m.insert("comptime", TokenTypeBase::Comptime);
    m.insert("import", TokenTypeBase::Import);
    m.insert("assert", TokenTypeBase::Assert);
    m.insert("enum", TokenTypeBase::Enum);
    m.insert("as", TokenTypeBase::As);
    m.insert("grad", TokenTypeBase::Grad);
    m.insert("vjp", TokenTypeBase::Vjp);
    m.insert("jvp", TokenTypeBase::Jvp);
    m.insert("requires", TokenTypeBase::Requires);
    m.insert("ensures", TokenTypeBase::Ensures);
    m.insert("invariant", TokenTypeBase::Invariant);
    m
});

pub struct Lexer<'a> {
    source: &'a str,
    iter: std::iter::Peekable<std::str::CharIndices<'a>>,
    line: usize,
    column: usize,
    pub preserve_comments: bool,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Self {
            source,
            iter: source.char_indices().peekable(),
            line: 1,
            column: 1,
            preserve_comments: false,
        }
    }

    pub fn new_with_comments(source: &'a str) -> Self {
        Self {
            source,
            iter: source.char_indices().peekable(),
            line: 1,
            column: 1,
            preserve_comments: true,
        }
    }

    fn advance(&mut self) -> Option<(usize, char)> {
        let res = self.iter.next()?;
        if res.1 == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(res)
    }

    fn peek(&mut self) -> Option<&(usize, char)> {
        self.iter.peek()
    }

    fn peek_char(&mut self) -> Option<char> {
        self.iter.peek().map(|&(_, c)| c)
    }

    fn current_byte_offset(&mut self) -> usize {
        self.peek()
            .map(|&(idx, _)| idx)
            .unwrap_or(self.source.len())
    }

    fn skip_whitespace(&mut self) {
        loop {
            let offset = self.current_byte_offset();
            if offset >= self.source.len() {
                break;
            }

            let mut chars = self.source[offset..].chars();
            if let Some(c) = chars.next() {
                if c.is_whitespace() {
                    self.advance();
                } else if self.source[offset..].starts_with("//") {
                    let rest = &self.source[offset..];
                    if rest.starts_with("///") && !rest.starts_with("////") {
                        break; // doc comment
                    }
                    // Line comment
                    self.advance(); // consume first '/'
                    self.advance(); // consume second '/'
                    while let Some((_, c)) = self.advance() {
                        if c == '\n' {
                            break;
                        }
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    fn identifier_or_keyword(&mut self, start_byte: usize, start_col: usize) -> Token<'a> {
        while let Some(c) = self.peek_char() {
            if c.is_alphanumeric() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }

        let end_byte = self.current_byte_offset();
        let text = &self.source[start_byte..end_byte];

        let kind = if let Some(k) = KEYWORDS.get(text) {
            k.clone()
        } else if text == "macro_rules" {
            if self.peek_char() == Some('!') {
                self.advance(); // consume '!'
            }
            TokenTypeBase::MacroRules
        } else {
            TokenTypeBase::Identifier(text)
        };

        TokenBase {
            kind,
            line: self.line,
            column: start_col,
            length: self.current_byte_offset() - start_byte,
        }
    }

    fn number(&mut self, start_byte: usize, start_col: usize) -> Token<'a> {
        while let Some(c) = self.peek_char() {
            if c.is_ascii_digit() || c.is_alphabetic() || c == '_' {
                self.advance();
            } else if c == '.' {
                let offset = self.current_byte_offset();
                let mut chars = self.source[offset + 1..].chars();
                if chars.next() == Some('.') {
                    break;
                }
                self.advance();
            } else {
                break;
            }
        }

        let end_byte = self.current_byte_offset();
        let text = &self.source[start_byte..end_byte];

        TokenBase {
            kind: TokenTypeBase::Number(text),
            line: self.line,
            column: start_col,
            length: self.current_byte_offset() - start_byte,
        }
    }

    fn string_literal(&mut self, start_byte: usize, start_col: usize) -> Token<'a> {
        let mut text: Option<String> = None;
        let mut raw_len = 0;
        while let Some(next_c) = self.peek_char() {
            if next_c == '"' {
                self.advance();
                break;
            }
            let mut char_to_push = self.advance().unwrap().1;
            if char_to_push == '\\' {
                if text.is_none() {
                    let mut initial_text = String::with_capacity(raw_len + 4);
                    initial_text.push_str(&self.source[start_byte + 1..start_byte + 1 + raw_len]);
                    text = Some(initial_text);
                }
                if let Some(esc_c) = self.peek_char() {
                    match esc_c {
                        'n' => {
                            self.advance();
                            char_to_push = '\n';
                        }
                        't' => {
                            self.advance();
                            char_to_push = '\t';
                        }
                        'r' => {
                            self.advance();
                            char_to_push = '\r';
                        }
                        '"' => {
                            self.advance();
                            char_to_push = '"';
                        }
                        '\\' => {
                            self.advance();
                            char_to_push = '\\';
                        }
                        _ => {
                            self.advance();
                            char_to_push = esc_c;
                        }
                    }
                }
            } else if text.is_none() {
                raw_len += char_to_push.len_utf8();
            }
            if let Some(ref mut t) = text {
                t.push(char_to_push);
            }
        }

        let end_byte = self.current_byte_offset();
        let literal = if let Some(t) = text {
            std::borrow::Cow::Owned(t)
        } else {
            std::borrow::Cow::Borrowed(&self.source[start_byte + 1..end_byte - 1])
        };

        TokenBase {
            kind: TokenTypeBase::StringLiteral(literal),
            line: self.line,
            column: start_col,
            length: self.current_byte_offset() - start_byte,
        }
    }

    fn lex_comment(&mut self, start_byte: usize, start_col: usize, is_doc: bool) -> TokenBase<'a> {
        while let Some(next_c) = self.peek_char() {
            if next_c == '\n' {
                break;
            }
            self.advance();
        }
        let end_byte = self.current_byte_offset();
        let comment = &self.source[start_byte..end_byte];
        TokenBase {
            kind: if is_doc {
                TokenTypeBase::DocComment(comment)
            } else {
                TokenTypeBase::Comment(comment)
            },
            line: self.line,
            column: start_col,
            length: end_byte - start_byte,
        }
    }

    pub fn next_token(&mut self) -> Token<'a> {
        if !self.preserve_comments {
            self.skip_whitespace();
        }

        let start_col = self.column;

        let (start_byte, c) = match self.peek() {
            Some(&(idx, c)) => (idx, c),
            None => {
                return TokenBase {
                    kind: TokenTypeBase::Eof,
                    line: self.line,
                    column: self.column,
                    length: 0,
                }
            }
        };

        if self.preserve_comments {
            if c.is_whitespace() {
                while let Some(next_c) = self.peek_char() {
                    if next_c.is_whitespace() {
                        self.advance();
                    } else {
                        break;
                    }
                }
                let end_byte = self.current_byte_offset();
                let ws = &self.source[start_byte..end_byte];
                return TokenBase {
                    kind: TokenTypeBase::Whitespace(ws),
                    line: self.line,
                    column: start_col,
                    length: self.current_byte_offset() - start_byte,
                };
            }

            if c == '/' {
                let rest = &self.source[self.current_byte_offset()..];
                if rest.starts_with("//") && !rest.starts_with("///") {
                    self.advance(); // consume first '/'
                    self.advance(); // consume second '/'
                    return self.lex_comment(start_byte, start_col, false);
                }
            }
        }

        if c == '/' {
            let rest = &self.source[self.current_byte_offset()..];
            if rest.starts_with("///") && !rest.starts_with("////") {
                self.advance(); // consume first '/'
                self.advance(); // consume second '/'
                self.advance(); // consume third '/'
                return self.lex_comment(start_byte, start_col, true);
            }
        }

        self.advance(); // consume c

        if c.is_alphabetic() || c == '_' {
            return self.identifier_or_keyword(start_byte, start_col);
        }

        if c.is_ascii_digit() {
            return self.number(start_byte, start_col);
        }

        if c == '"' {
            return self.string_literal(start_byte, start_col);
        }

        let kind = match c {
            '(' => TokenTypeBase::LeftParen,
            ')' => TokenTypeBase::RightParen,
            '{' => TokenTypeBase::LeftBrace,
            '}' => TokenTypeBase::RightBrace,
            '[' => TokenTypeBase::LeftBracket,
            ']' => TokenTypeBase::RightBracket,

            ';' => TokenTypeBase::Semicolon,
            ',' => TokenTypeBase::Comma,
            '+' => {
                if self.peek_char() == Some('=') {
                    self.advance();
                    TokenTypeBase::PlusEquals
                } else {
                    TokenTypeBase::Plus
                }
            }
            '*' => TokenTypeBase::Star,
            '@' => TokenTypeBase::At,
            '$' => TokenTypeBase::Dollar,
            '/' => TokenTypeBase::Slash,
            '=' => {
                if self.peek_char() == Some('=') {
                    self.advance();
                    TokenTypeBase::EqEq
                } else if self.peek_char() == Some('>') {
                    self.advance();
                    TokenTypeBase::FatArrow
                } else {
                    TokenTypeBase::Equals
                }
            }
            '!' => {
                if self.peek_char() == Some('=') {
                    self.advance();
                    TokenTypeBase::NotEq
                } else {
                    TokenTypeBase::Bang
                }
            }
            '<' => {
                if self.peek_char() == Some('=') {
                    self.advance();
                    TokenTypeBase::LessEq
                } else {
                    TokenTypeBase::LeftAngle
                }
            }
            '>' => {
                if self.peek_char() == Some('=') {
                    self.advance();
                    TokenTypeBase::GreaterEq
                } else {
                    TokenTypeBase::RightAngle
                }
            }
            '&' => {
                if self.peek_char() == Some('&') {
                    self.advance();
                    TokenTypeBase::AndAnd
                } else {
                    TokenTypeBase::Ampersand
                }
            }
            '|' => {
                if self.peek_char() == Some('|') {
                    self.advance();
                    TokenTypeBase::OrOr
                } else {
                    TokenTypeBase::Pipe
                }
            }
            '.' => {
                if self.peek_char() == Some('.') {
                    self.advance();
                    TokenTypeBase::DoubleDot
                } else {
                    TokenTypeBase::Dot
                }
            }
            '-' => {
                if self.peek_char() == Some('>') {
                    self.advance();
                    TokenTypeBase::Arrow
                } else {
                    TokenTypeBase::Minus
                }
            }
            ':' => {
                if self.peek_char() == Some(':') {
                    self.advance();
                    TokenTypeBase::DoubleColon
                } else {
                    TokenTypeBase::Colon
                }
            }
            _ => TokenTypeBase::Unknown(c),
        };

        TokenBase {
            kind,
            line: self.line,
            column: start_col,
            length: self.current_byte_offset() - start_byte,
        }
    }

    pub fn tokenize(&mut self) -> Vec<Token<'a>> {
        let mut tokens = Vec::new();
        loop {
            let t = self.next_token();
            let is_eof = t.kind == TokenTypeBase::Eof;
            tokens.push(t);
            if is_eof {
                break;
            }
        }
        tokens
    }
}

impl<'a> Token<'a> {
    pub fn into_owned(self) -> OwnedToken {
        let kind = match self.kind {
            TokenTypeBase::Fn => TokenTypeBase::Fn,
            TokenTypeBase::Let => TokenTypeBase::Let,
            TokenTypeBase::Mut => TokenTypeBase::Mut,
            TokenTypeBase::For => TokenTypeBase::For,
            TokenTypeBase::In => TokenTypeBase::In,
            TokenTypeBase::If => TokenTypeBase::If,
            TokenTypeBase::Else => TokenTypeBase::Else,
            TokenTypeBase::Loop => TokenTypeBase::Loop,
            TokenTypeBase::Break => TokenTypeBase::Break,
            TokenTypeBase::Continue => TokenTypeBase::Continue,
            TokenTypeBase::Return => TokenTypeBase::Return,
            TokenTypeBase::Spawn => TokenTypeBase::Spawn,
            TokenTypeBase::On => TokenTypeBase::On,
            TokenTypeBase::Transfer => TokenTypeBase::Transfer,
            TokenTypeBase::Unroll => TokenTypeBase::Unroll,
            TokenTypeBase::Across => TokenTypeBase::Across,
            TokenTypeBase::Match => TokenTypeBase::Match,
            TokenTypeBase::Struct => TokenTypeBase::Struct,
            TokenTypeBase::Unsafe => TokenTypeBase::Unsafe,
            TokenTypeBase::Safe => TokenTypeBase::Safe,
            TokenTypeBase::Extern => TokenTypeBase::Extern,
            TokenTypeBase::Trait => TokenTypeBase::Trait,
            TokenTypeBase::Impl => TokenTypeBase::Impl,
            TokenTypeBase::Comptime => TokenTypeBase::Comptime,
            TokenTypeBase::Import => TokenTypeBase::Import,
            TokenTypeBase::Assert => TokenTypeBase::Assert,
            TokenTypeBase::Enum => TokenTypeBase::Enum,
            TokenTypeBase::As => TokenTypeBase::As,
            TokenTypeBase::Grad => TokenTypeBase::Grad,
            TokenTypeBase::Vjp => TokenTypeBase::Vjp,
            TokenTypeBase::Jvp => TokenTypeBase::Jvp,
            TokenTypeBase::Requires => TokenTypeBase::Requires,
            TokenTypeBase::Ensures => TokenTypeBase::Ensures,
            TokenTypeBase::Invariant => TokenTypeBase::Invariant,
            TokenTypeBase::MacroRules => TokenTypeBase::MacroRules,
            TokenTypeBase::Topology => TokenTypeBase::Topology,
            TokenTypeBase::Memory => TokenTypeBase::Memory,
            TokenTypeBase::Ref => TokenTypeBase::Ref,
            TokenTypeBase::Verified => TokenTypeBase::Verified,
            TokenTypeBase::Pinned => TokenTypeBase::Pinned,
            TokenTypeBase::HardwareState => TokenTypeBase::HardwareState,
            TokenTypeBase::Identifier(s) => {
                TokenTypeBase::Identifier(crate::symbol::Symbol::from(s))
            }
            TokenTypeBase::Number(s) => TokenTypeBase::Number(crate::symbol::Symbol::from(s)),
            TokenTypeBase::StringLiteral(s) => TokenTypeBase::StringLiteral(s.into_owned()),
            TokenTypeBase::LeftParen => TokenTypeBase::LeftParen,
            TokenTypeBase::RightParen => TokenTypeBase::RightParen,
            TokenTypeBase::LeftBrace => TokenTypeBase::LeftBrace,
            TokenTypeBase::RightBrace => TokenTypeBase::RightBrace,
            TokenTypeBase::LeftBracket => TokenTypeBase::LeftBracket,
            TokenTypeBase::RightBracket => TokenTypeBase::RightBracket,
            TokenTypeBase::LeftAngle => TokenTypeBase::LeftAngle,
            TokenTypeBase::RightAngle => TokenTypeBase::RightAngle,
            TokenTypeBase::Colon => TokenTypeBase::Colon,
            TokenTypeBase::DoubleColon => TokenTypeBase::DoubleColon,
            TokenTypeBase::Semicolon => TokenTypeBase::Semicolon,
            TokenTypeBase::Comma => TokenTypeBase::Comma,
            TokenTypeBase::Equals => TokenTypeBase::Equals,
            TokenTypeBase::PlusEquals => TokenTypeBase::PlusEquals,
            TokenTypeBase::Arrow => TokenTypeBase::Arrow,
            TokenTypeBase::FatArrow => TokenTypeBase::FatArrow,
            TokenTypeBase::Plus => TokenTypeBase::Plus,
            TokenTypeBase::Minus => TokenTypeBase::Minus,
            TokenTypeBase::Star => TokenTypeBase::Star,
            TokenTypeBase::Slash => TokenTypeBase::Slash,
            TokenTypeBase::Dot => TokenTypeBase::Dot,
            TokenTypeBase::DoubleDot => TokenTypeBase::DoubleDot,
            TokenTypeBase::Ampersand => TokenTypeBase::Ampersand,
            TokenTypeBase::At => TokenTypeBase::At,
            TokenTypeBase::Dollar => TokenTypeBase::Dollar,
            TokenTypeBase::EqEq => TokenTypeBase::EqEq,
            TokenTypeBase::NotEq => TokenTypeBase::NotEq,
            TokenTypeBase::LessEq => TokenTypeBase::LessEq,
            TokenTypeBase::GreaterEq => TokenTypeBase::GreaterEq,
            TokenTypeBase::AndAnd => TokenTypeBase::AndAnd,
            TokenTypeBase::OrOr => TokenTypeBase::OrOr,
            TokenTypeBase::Bang => TokenTypeBase::Bang,
            TokenTypeBase::Pipe => TokenTypeBase::Pipe,
            TokenTypeBase::Eof => TokenTypeBase::Eof,
            TokenTypeBase::Unknown(c) => TokenTypeBase::Unknown(c),
            TokenTypeBase::Comment(s) => TokenTypeBase::Comment(crate::symbol::Symbol::from(s)),
            TokenTypeBase::DocComment(s) => {
                TokenTypeBase::DocComment(crate::symbol::Symbol::from(s))
            }
            TokenTypeBase::Whitespace(s) => {
                TokenTypeBase::Whitespace(crate::symbol::Symbol::from(s))
            }
        };
        OwnedToken {
            kind,
            line: self.line,
            column: self.column,
            length: self.length,
        }
    }
}

impl OwnedToken {
    pub fn as_token<'a>(&'a self) -> Token<'a> {
        let kind = match &self.kind {
            TokenTypeBase::Fn => TokenTypeBase::Fn,
            TokenTypeBase::Let => TokenTypeBase::Let,
            TokenTypeBase::Mut => TokenTypeBase::Mut,
            TokenTypeBase::For => TokenTypeBase::For,
            TokenTypeBase::In => TokenTypeBase::In,
            TokenTypeBase::If => TokenTypeBase::If,
            TokenTypeBase::Else => TokenTypeBase::Else,
            TokenTypeBase::Loop => TokenTypeBase::Loop,
            TokenTypeBase::Break => TokenTypeBase::Break,
            TokenTypeBase::Continue => TokenTypeBase::Continue,
            TokenTypeBase::Return => TokenTypeBase::Return,
            TokenTypeBase::Spawn => TokenTypeBase::Spawn,
            TokenTypeBase::On => TokenTypeBase::On,
            TokenTypeBase::Transfer => TokenTypeBase::Transfer,
            TokenTypeBase::Unroll => TokenTypeBase::Unroll,
            TokenTypeBase::Across => TokenTypeBase::Across,
            TokenTypeBase::Match => TokenTypeBase::Match,
            TokenTypeBase::Struct => TokenTypeBase::Struct,
            TokenTypeBase::Unsafe => TokenTypeBase::Unsafe,
            TokenTypeBase::Safe => TokenTypeBase::Safe,
            TokenTypeBase::Extern => TokenTypeBase::Extern,
            TokenTypeBase::Trait => TokenTypeBase::Trait,
            TokenTypeBase::Impl => TokenTypeBase::Impl,
            TokenTypeBase::Comptime => TokenTypeBase::Comptime,
            TokenTypeBase::Import => TokenTypeBase::Import,
            TokenTypeBase::Assert => TokenTypeBase::Assert,
            TokenTypeBase::Enum => TokenTypeBase::Enum,
            TokenTypeBase::As => TokenTypeBase::As,
            TokenTypeBase::Grad => TokenTypeBase::Grad,
            TokenTypeBase::Vjp => TokenTypeBase::Vjp,
            TokenTypeBase::Jvp => TokenTypeBase::Jvp,
            TokenTypeBase::Requires => TokenTypeBase::Requires,
            TokenTypeBase::Ensures => TokenTypeBase::Ensures,
            TokenTypeBase::Invariant => TokenTypeBase::Invariant,
            TokenTypeBase::MacroRules => TokenTypeBase::MacroRules,
            TokenTypeBase::Topology => TokenTypeBase::Topology,
            TokenTypeBase::Memory => TokenTypeBase::Memory,
            TokenTypeBase::Ref => TokenTypeBase::Ref,
            TokenTypeBase::Verified => TokenTypeBase::Verified,
            TokenTypeBase::Pinned => TokenTypeBase::Pinned,
            TokenTypeBase::HardwareState => TokenTypeBase::HardwareState,
            TokenTypeBase::Identifier(s) => TokenTypeBase::Identifier(&**s),
            TokenTypeBase::Number(s) => TokenTypeBase::Number(&**s),
            TokenTypeBase::StringLiteral(s) => {
                TokenTypeBase::StringLiteral(std::borrow::Cow::Borrowed(s.as_str()))
            }
            TokenTypeBase::LeftParen => TokenTypeBase::LeftParen,
            TokenTypeBase::RightParen => TokenTypeBase::RightParen,
            TokenTypeBase::LeftBrace => TokenTypeBase::LeftBrace,
            TokenTypeBase::RightBrace => TokenTypeBase::RightBrace,
            TokenTypeBase::LeftBracket => TokenTypeBase::LeftBracket,
            TokenTypeBase::RightBracket => TokenTypeBase::RightBracket,
            TokenTypeBase::LeftAngle => TokenTypeBase::LeftAngle,
            TokenTypeBase::RightAngle => TokenTypeBase::RightAngle,
            TokenTypeBase::Colon => TokenTypeBase::Colon,
            TokenTypeBase::DoubleColon => TokenTypeBase::DoubleColon,
            TokenTypeBase::Semicolon => TokenTypeBase::Semicolon,
            TokenTypeBase::Comma => TokenTypeBase::Comma,
            TokenTypeBase::Equals => TokenTypeBase::Equals,
            TokenTypeBase::PlusEquals => TokenTypeBase::PlusEquals,
            TokenTypeBase::Arrow => TokenTypeBase::Arrow,
            TokenTypeBase::FatArrow => TokenTypeBase::FatArrow,
            TokenTypeBase::Plus => TokenTypeBase::Plus,
            TokenTypeBase::Minus => TokenTypeBase::Minus,
            TokenTypeBase::Star => TokenTypeBase::Star,
            TokenTypeBase::Slash => TokenTypeBase::Slash,
            TokenTypeBase::Dot => TokenTypeBase::Dot,
            TokenTypeBase::DoubleDot => TokenTypeBase::DoubleDot,
            TokenTypeBase::Ampersand => TokenTypeBase::Ampersand,
            TokenTypeBase::At => TokenTypeBase::At,
            TokenTypeBase::Dollar => TokenTypeBase::Dollar,
            TokenTypeBase::EqEq => TokenTypeBase::EqEq,
            TokenTypeBase::NotEq => TokenTypeBase::NotEq,
            TokenTypeBase::LessEq => TokenTypeBase::LessEq,
            TokenTypeBase::GreaterEq => TokenTypeBase::GreaterEq,
            TokenTypeBase::AndAnd => TokenTypeBase::AndAnd,
            TokenTypeBase::OrOr => TokenTypeBase::OrOr,
            TokenTypeBase::Bang => TokenTypeBase::Bang,
            TokenTypeBase::Pipe => TokenTypeBase::Pipe,
            TokenTypeBase::Eof => TokenTypeBase::Eof,
            TokenTypeBase::Unknown(c) => TokenTypeBase::Unknown(*c),
            TokenTypeBase::Comment(s) => TokenTypeBase::Comment(&**s),
            TokenTypeBase::DocComment(s) => TokenTypeBase::DocComment(&**s),
            TokenTypeBase::Whitespace(s) => TokenTypeBase::Whitespace(&**s),
        };
        Token {
            kind,
            line: self.line,
            column: self.column,
            length: self.length,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(input: &str) -> Vec<TokenType<'_>> {
        let mut lexer = Lexer::new(input);
        lexer
            .tokenize()
            .into_iter()
            .map(|t| t.kind)
            .filter(|k| *k != TokenTypeBase::Eof)
            .collect()
    }

    fn lex_first(input: &str) -> TokenType<'_> {
        let mut lexer = Lexer::new(input);
        lexer.tokenize().into_iter().next().unwrap().kind
    }

    #[test]
    fn test_lex_keywords() {
        assert_eq!(lex_first("fn"), TokenTypeBase::Fn);
        assert_eq!(lex_first("let"), TokenTypeBase::Let);
        assert_eq!(lex_first("mut"), TokenTypeBase::Mut);
        assert_eq!(lex_first("for"), TokenTypeBase::For);
        assert_eq!(lex_first("in"), TokenTypeBase::In);
        assert_eq!(lex_first("if"), TokenTypeBase::If);
        assert_eq!(lex_first("else"), TokenTypeBase::Else);
        assert_eq!(lex_first("return"), TokenTypeBase::Return);
        assert_eq!(lex_first("match"), TokenTypeBase::Match);
        assert_eq!(lex_first("struct"), TokenTypeBase::Struct);
        assert_eq!(lex_first("enum"), TokenTypeBase::Enum);
        assert_eq!(lex_first("unsafe"), TokenTypeBase::Unsafe);
        assert_eq!(lex_first("extern"), TokenTypeBase::Extern);
        assert_eq!(lex_first("trait"), TokenTypeBase::Trait);
        assert_eq!(lex_first("impl"), TokenTypeBase::Impl);
        assert_eq!(lex_first("import"), TokenTypeBase::Import);
        assert_eq!(lex_first("grad"), TokenTypeBase::Grad);
        assert_eq!(lex_first("vjp"), TokenTypeBase::Vjp);
        assert_eq!(lex_first("jvp"), TokenTypeBase::Jvp);
    }

    #[test]
    fn test_lex_operators() {
        assert_eq!(lex_first("+"), TokenTypeBase::Plus);
        assert_eq!(lex_first("-"), TokenTypeBase::Minus);
        assert_eq!(lex_first("*"), TokenTypeBase::Star);
        assert_eq!(lex_first("/"), TokenTypeBase::Slash);
        assert_eq!(lex_first("@"), TokenTypeBase::At);
        assert_eq!(lex_first("("), TokenTypeBase::LeftParen);
        assert_eq!(lex_first(")"), TokenTypeBase::RightParen);
        assert_eq!(lex_first("{"), TokenTypeBase::LeftBrace);
        assert_eq!(lex_first("}"), TokenTypeBase::RightBrace);
        assert_eq!(lex_first("["), TokenTypeBase::LeftBracket);
        assert_eq!(lex_first("]"), TokenTypeBase::RightBracket);
        assert_eq!(lex_first(";"), TokenTypeBase::Semicolon);
        assert_eq!(lex_first(","), TokenTypeBase::Comma);
        assert_eq!(lex_first("&"), TokenTypeBase::Ampersand);
        assert_eq!(lex_first("!"), TokenTypeBase::Bang);
        assert_eq!(lex_first("."), TokenTypeBase::Dot);
        assert_eq!(lex_first(":"), TokenTypeBase::Colon);
    }

    #[test]
    fn test_lex_multi_char_operators() {
        assert_eq!(lex_first("=="), TokenTypeBase::EqEq);
        assert_eq!(lex_first("!="), TokenTypeBase::NotEq);
        assert_eq!(lex_first("<="), TokenTypeBase::LessEq);
        assert_eq!(lex_first(">="), TokenTypeBase::GreaterEq);
        assert_eq!(lex_first("&&"), TokenTypeBase::AndAnd);
        assert_eq!(lex_first("||"), TokenTypeBase::OrOr);
        assert_eq!(lex_first("::"), TokenTypeBase::DoubleColon);
        assert_eq!(lex_first(".."), TokenTypeBase::DoubleDot);
        assert_eq!(lex_first("->"), TokenTypeBase::Arrow);
        assert_eq!(lex_first("=>"), TokenTypeBase::FatArrow);
        assert_eq!(lex_first("+="), TokenTypeBase::PlusEquals);
    }

    #[test]
    fn test_lex_number_integer() {
        assert_eq!(lex_first("42"), TokenTypeBase::Number("42"));
    }

    #[test]
    fn test_lex_number_float() {
        assert_eq!(lex_first("3.14"), TokenTypeBase::Number("3.14"));
    }

    #[test]
    fn test_lex_number_with_suffix() {
        assert_eq!(lex_first("42i64"), TokenTypeBase::Number("42i64"));
    }

    #[test]
    fn test_lex_string_literal() {
        let kind = lex_first("\"hello\"");
        if let TokenTypeBase::StringLiteral(s) = kind {
            assert_eq!(s.as_ref(), "hello");
        } else {
            panic!("Expected StringLiteral, got {:?}", kind);
        }
    }

    #[test]
    fn test_lex_identifier() {
        assert_eq!(lex_first("my_var"), TokenTypeBase::Identifier("my_var"));
    }

    #[test]
    fn test_lex_identifier_vs_keyword() {
        // "fn" is a keyword, "fn_name" is an identifier
        assert_eq!(lex_first("fn"), TokenTypeBase::Fn);
        assert_eq!(lex_first("fn_name"), TokenTypeBase::Identifier("fn_name"));
    }

    #[test]
    fn test_lex_empty_input() {
        let mut lexer = Lexer::new("");
        let tokens = lexer.tokenize();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TokenTypeBase::Eof);
    }

    #[test]
    fn test_lex_line_tracking() {
        let input = "a\nb";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        // Filter out Eof
        let non_eof: Vec<_> = tokens
            .iter()
            .filter(|t| t.kind != TokenTypeBase::Eof)
            .collect();
        assert_eq!(non_eof.len(), 2);
        assert_eq!(non_eof[0].line, 1);
        assert_eq!(non_eof[1].line, 2);
    }

    #[test]
    fn test_lex_comment_skipped_by_default() {
        let input = "a // comment\nb";
        let kinds = lex(input);
        // Comments should be skipped by default
        assert_eq!(kinds.len(), 2);
        assert_eq!(kinds[0], TokenTypeBase::Identifier("a"));
        assert_eq!(kinds[1], TokenTypeBase::Identifier("b"));
    }

    #[test]
    fn test_lex_comment_preserved() {
        let input = "a // comment";
        let mut lexer = Lexer::new_with_comments(input);
        let tokens = lexer.tokenize();
        let has_comment = tokens
            .iter()
            .any(|t| matches!(&t.kind, TokenTypeBase::Comment(_)));
        assert!(
            has_comment,
            "Expected comment token when preserve_comments is true"
        );
    }

    #[test]
    fn test_lex_multiple_expressions() {
        let kinds = lex("a + b * c");
        assert_eq!(kinds.len(), 5);
        assert_eq!(kinds[0], TokenTypeBase::Identifier("a"));
        assert_eq!(kinds[1], TokenTypeBase::Plus);
        assert_eq!(kinds[2], TokenTypeBase::Identifier("b"));
        assert_eq!(kinds[3], TokenTypeBase::Star);
        assert_eq!(kinds[4], TokenTypeBase::Identifier("c"));
    }

    #[test]
    fn test_lex_function_signature() {
        let kinds = lex("fn foo(x: i32) -> f32");
        assert_eq!(kinds[0], TokenTypeBase::Fn);
        assert_eq!(kinds[1], TokenTypeBase::Identifier("foo"));
        assert_eq!(kinds[2], TokenTypeBase::LeftParen);
        assert_eq!(kinds[3], TokenTypeBase::Identifier("x"));
        assert_eq!(kinds[4], TokenTypeBase::Colon);
        assert_eq!(kinds[5], TokenTypeBase::Identifier("i32"));
        assert_eq!(kinds[6], TokenTypeBase::RightParen);
        assert_eq!(kinds[7], TokenTypeBase::Arrow);
        assert_eq!(kinds[8], TokenTypeBase::Identifier("f32"));
    }
}
