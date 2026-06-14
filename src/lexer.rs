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
use std::borrow::Cow;

#[derive(Debug, PartialEq, Clone)]
pub enum TokenType<'a> {
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
    Identifier(&'a str),
    Number(&'a str),
    StringLiteral(Cow<'a, str>),

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
    Comment(&'a str),
    Whitespace(&'a str),
}

impl<'a> std::fmt::Display for TokenType<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenType::Fn => write!(f, "fn"),
            TokenType::Let => write!(f, "let"),
            TokenType::Mut => write!(f, "mut"),
            TokenType::For => write!(f, "for"),
            TokenType::In => write!(f, "in"),
            TokenType::If => write!(f, "if"),
            TokenType::Else => write!(f, "else"),
            TokenType::Loop => write!(f, "loop"),
            TokenType::Break => write!(f, "break"),
            TokenType::Continue => write!(f, "continue"),
            TokenType::Return => write!(f, "return"),
            TokenType::Spawn => write!(f, "spawn"),
            TokenType::On => write!(f, "on"),
            TokenType::Transfer => write!(f, "transfer"),
            TokenType::Unroll => write!(f, "unroll"),
            TokenType::Across => write!(f, "across"),
            TokenType::Match => write!(f, "match"),
            TokenType::Struct => write!(f, "struct"),
            TokenType::Unsafe => write!(f, "unsafe"),
            TokenType::Safe => write!(f, "safe"),
            TokenType::Extern => write!(f, "extern"),
            TokenType::Trait => write!(f, "trait"),
            TokenType::Impl => write!(f, "impl"),
            TokenType::Comptime => write!(f, "comptime"),
            TokenType::Import => write!(f, "import"),
            TokenType::Assert => write!(f, "assert"),
            TokenType::Enum => write!(f, "enum"),
            TokenType::As => write!(f, "as"),
            TokenType::Grad => write!(f, "grad"),
            TokenType::Vjp => write!(f, "vjp"),
            TokenType::Jvp => write!(f, "jvp"),
            TokenType::Requires => write!(f, "requires"),
            TokenType::Ensures => write!(f, "ensures"),
            TokenType::Invariant => write!(f, "invariant"),

            TokenType::Topology => write!(f, "Topology"),
            TokenType::Memory => write!(f, "Memory"),
            TokenType::Ref => write!(f, "Ref"),
            TokenType::Verified => write!(f, "Verified"),
            TokenType::Pinned => write!(f, "Pinned"),
            TokenType::HardwareState => write!(f, "HardwareState"),

            TokenType::MacroRules => write!(f, "macro_rules"),

            TokenType::Identifier(s) => write!(f, "{}", s),
            TokenType::Number(s) => write!(f, "{}", s),
            TokenType::StringLiteral(s) => write!(f, "\"{}\"", s),

            TokenType::LeftParen => write!(f, "("),
            TokenType::RightParen => write!(f, ")"),
            TokenType::LeftBrace => write!(f, "{{"),
            TokenType::RightBrace => write!(f, "}}"),
            TokenType::LeftBracket => write!(f, "["),
            TokenType::RightBracket => write!(f, "]"),
            TokenType::LeftAngle => write!(f, "<"),
            TokenType::RightAngle => write!(f, ">"),
            TokenType::Colon => write!(f, ":"),
            TokenType::DoubleColon => write!(f, "::"),
            TokenType::Semicolon => write!(f, ";"),
            TokenType::Comma => write!(f, ","),
            TokenType::Equals => write!(f, "="),
            TokenType::PlusEquals => write!(f, "+="),
            TokenType::Arrow => write!(f, "->"),
            TokenType::FatArrow => write!(f, "=>"),
            TokenType::Plus => write!(f, "+"),
            TokenType::Minus => write!(f, "-"),
            TokenType::Star => write!(f, "*"),
            TokenType::Slash => write!(f, "/"),
            TokenType::Dot => write!(f, "."),
            TokenType::DoubleDot => write!(f, ".."),
            TokenType::Ampersand => write!(f, "&"),
            TokenType::At => write!(f, "@"),
            TokenType::Dollar => write!(f, "$"),

            TokenType::EqEq => write!(f, "=="),
            TokenType::NotEq => write!(f, "!="),
            TokenType::LessEq => write!(f, "<="),
            TokenType::GreaterEq => write!(f, ">="),
            TokenType::AndAnd => write!(f, "&&"),
            TokenType::OrOr => write!(f, "||"),
            TokenType::Bang => write!(f, "!"),
            TokenType::Pipe => write!(f, "|"),

            TokenType::Comment(s) => write!(f, "{}", s),
            TokenType::Whitespace(s) => write!(f, "{}", s),
            TokenType::Unknown(c) => write!(f, "{}", c),
            TokenType::Eof => write!(f, ""),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token<'a> {
    pub kind: TokenType<'a>,
    pub line: usize,
    pub column: usize,
    pub length: usize,
}

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
        while let Some(&(_idx, c)) = self.peek() {
            if c.is_whitespace() {
                self.advance();
            } else if c == '/' {
                // Peek ahead to check for comments
                let mut temp = self.iter.clone();
                temp.next(); // consume '/'
                if let Some(&(_, '/')) = temp.peek() {
                    // Line comment
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

        let kind = match text {
            "fn" => TokenType::Fn,
            "let" => TokenType::Let,
            "mut" => TokenType::Mut,
            "for" => TokenType::For,
            "in" => TokenType::In,
            "if" => TokenType::If,
            "else" => TokenType::Else,
            "loop" => TokenType::Loop,
            "break" => TokenType::Break,
            "continue" => TokenType::Continue,
            "return" => TokenType::Return,
            "spawn" => TokenType::Spawn,
            "on" => TokenType::On,
            "transfer" => TokenType::Transfer,
            "unroll" => TokenType::Unroll,
            "across" => TokenType::Across,
            "match" => TokenType::Match,
            "Topology" => TokenType::Topology,
            "Memory" => TokenType::Memory,
            "Ref" => TokenType::Ref,
            "Verified" => TokenType::Verified,
            "Pinned" => TokenType::Pinned,
            "HardwareState" => TokenType::HardwareState,
            "struct" => TokenType::Struct,
            "unsafe" => TokenType::Unsafe,
            "safe" => TokenType::Safe,
            "extern" => TokenType::Extern,
            "trait" => TokenType::Trait,
            "impl" => TokenType::Impl,
            "comptime" => TokenType::Comptime,
            "import" => TokenType::Import,
            "assert" => TokenType::Assert,
            "enum" => TokenType::Enum,
            "as" => TokenType::As,
            "grad" => TokenType::Grad,
            "vjp" => TokenType::Vjp,
            "jvp" => TokenType::Jvp,
            "requires" => TokenType::Requires,
            "ensures" => TokenType::Ensures,
            "invariant" => TokenType::Invariant,
            "macro_rules" => {
                if self.peek_char() == Some('!') {
                    self.advance(); // consume '!'
                }
                TokenType::MacroRules
            }
            _ => TokenType::Identifier(text),
        };

        Token {
            kind,
            line: self.line,
            column: start_col,
            length: self.column - start_col,
        }
    }

    fn number(&mut self, start_byte: usize, start_col: usize) -> Token<'a> {
        while let Some(c) = self.peek_char() {
            if c.is_ascii_digit() || c.is_alphabetic() || c == '_' {
                self.advance();
            } else if c == '.' {
                let mut temp = self.iter.clone();
                temp.next();
                if let Some(&(_, '.')) = temp.peek() {
                    break;
                }
                self.advance();
            } else {
                break;
            }
        }

        let end_byte = self.current_byte_offset();
        let text = &self.source[start_byte..end_byte];

        Token {
            kind: TokenType::Number(text),
            line: self.line,
            column: start_col,
            length: self.column - start_col,
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
                return Token {
                    kind: TokenType::Eof,
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
                return Token {
                    kind: TokenType::Whitespace(ws),
                    line: self.line,
                    column: start_col,
                    length: self.column - start_col,
                };
            }

            if c == '/' {
                let mut temp = self.iter.clone();
                temp.next();
                if let Some(&(_, '/')) = temp.peek() {
                    self.advance(); // '/'
                    self.advance(); // '/'
                    while let Some(next_c) = self.peek_char() {
                        if next_c == '\n' {
                            break;
                        }
                        self.advance();
                    }
                    let end_byte = self.current_byte_offset();
                    let comment = &self.source[start_byte..end_byte];
                    return Token {
                        kind: TokenType::Comment(comment),
                        line: self.line,
                        column: start_col,
                        length: self.column - start_col,
                    };
                }
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
            let mut text = String::new();
            let mut has_escapes = false;
            while let Some(next_c) = self.peek_char() {
                if next_c == '"' {
                    self.advance();
                    break;
                }
                let mut char_to_push = self.advance().unwrap().1;
                if char_to_push == '\\' {
                    has_escapes = true;
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
                            _ => {}
                        }
                    }
                }
                if has_escapes {
                    text.push(char_to_push);
                }
            }

            let end_byte = self.current_byte_offset();
            let raw_text = &self.source[start_byte + 1..end_byte - 1]; // exclude quotes

            let literal = if has_escapes {
                Cow::Owned(text)
            } else {
                Cow::Borrowed(raw_text)
            };

            return Token {
                kind: TokenType::StringLiteral(literal),
                line: self.line,
                column: start_col,
                length: self.column - start_col,
            };
        }

        let kind = match c {
            '(' => TokenType::LeftParen,
            ')' => TokenType::RightParen,
            '{' => TokenType::LeftBrace,
            '}' => TokenType::RightBrace,
            '[' => TokenType::LeftBracket,
            ']' => TokenType::RightBracket,

            ';' => TokenType::Semicolon,
            ',' => TokenType::Comma,
            '+' => {
                if self.peek_char() == Some('=') {
                    self.advance();
                    TokenType::PlusEquals
                } else {
                    TokenType::Plus
                }
            }
            '*' => TokenType::Star,
            '@' => TokenType::At,
            '$' => TokenType::Dollar,
            '/' => TokenType::Slash,
            '=' => {
                if self.peek_char() == Some('=') {
                    self.advance();
                    TokenType::EqEq
                } else if self.peek_char() == Some('>') {
                    self.advance();
                    TokenType::FatArrow
                } else {
                    TokenType::Equals
                }
            }
            '!' => {
                if self.peek_char() == Some('=') {
                    self.advance();
                    TokenType::NotEq
                } else {
                    TokenType::Bang
                }
            }
            '<' => {
                if self.peek_char() == Some('=') {
                    self.advance();
                    TokenType::LessEq
                } else {
                    TokenType::LeftAngle
                }
            }
            '>' => {
                if self.peek_char() == Some('=') {
                    self.advance();
                    TokenType::GreaterEq
                } else {
                    TokenType::RightAngle
                }
            }
            '&' => {
                if self.peek_char() == Some('&') {
                    self.advance();
                    TokenType::AndAnd
                } else {
                    TokenType::Ampersand
                }
            }
            '|' => {
                if self.peek_char() == Some('|') {
                    self.advance();
                    TokenType::OrOr
                } else {
                    TokenType::Pipe
                }
            }
            '.' => {
                if self.peek_char() == Some('.') {
                    self.advance();
                    TokenType::DoubleDot
                } else {
                    TokenType::Dot
                }
            }
            '-' => {
                if self.peek_char() == Some('>') {
                    self.advance();
                    TokenType::Arrow
                } else {
                    TokenType::Minus
                }
            }
            ':' => {
                if self.peek_char() == Some(':') {
                    self.advance();
                    TokenType::DoubleColon
                } else {
                    TokenType::Colon
                }
            }
            _ => TokenType::Unknown(c),
        };

        Token {
            kind,
            line: self.line,
            column: start_col,
            length: self.column - start_col,
        }
    }

    pub fn tokenize(&mut self) -> Vec<Token<'a>> {
        let mut tokens = Vec::new();
        loop {
            let t = self.next_token();
            let is_eof = t.kind == TokenType::Eof;
            tokens.push(t);
            if is_eof {
                break;
            }
        }
        tokens
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum OwnedTokenType {
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
    Topology,
    Memory,
    Ref,
    Verified,
    Pinned,
    HardwareState,
    Identifier(String),
    Number(String),
    StringLiteral(String),
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
    EqEq,
    NotEq,
    LessEq,
    GreaterEq,
    AndAnd,
    OrOr,
    Bang,
    Pipe,
    Eof,
    Unknown(char),
    Comment(String),
    Whitespace(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct OwnedToken {
    pub kind: OwnedTokenType,
    pub line: usize,
    pub column: usize,
    pub length: usize,
}

impl<'a> Token<'a> {
    pub fn into_owned(self) -> OwnedToken {
        let kind = match self.kind {
            TokenType::Fn => OwnedTokenType::Fn,
            TokenType::Let => OwnedTokenType::Let,
            TokenType::Mut => OwnedTokenType::Mut,
            TokenType::For => OwnedTokenType::For,
            TokenType::In => OwnedTokenType::In,
            TokenType::If => OwnedTokenType::If,
            TokenType::Else => OwnedTokenType::Else,
            TokenType::Loop => OwnedTokenType::Loop,
            TokenType::Break => OwnedTokenType::Break,
            TokenType::Continue => OwnedTokenType::Continue,
            TokenType::Return => OwnedTokenType::Return,
            TokenType::Spawn => OwnedTokenType::Spawn,
            TokenType::On => OwnedTokenType::On,
            TokenType::Transfer => OwnedTokenType::Transfer,
            TokenType::Unroll => OwnedTokenType::Unroll,
            TokenType::Across => OwnedTokenType::Across,
            TokenType::Match => OwnedTokenType::Match,
            TokenType::Struct => OwnedTokenType::Struct,
            TokenType::Unsafe => OwnedTokenType::Unsafe,
            TokenType::Safe => OwnedTokenType::Safe,
            TokenType::Extern => OwnedTokenType::Extern,
            TokenType::Trait => OwnedTokenType::Trait,
            TokenType::Impl => OwnedTokenType::Impl,
            TokenType::Comptime => OwnedTokenType::Comptime,
            TokenType::Import => OwnedTokenType::Import,
            TokenType::Assert => OwnedTokenType::Assert,
            TokenType::Enum => OwnedTokenType::Enum,
            TokenType::As => OwnedTokenType::As,
            TokenType::Grad => OwnedTokenType::Grad,
            TokenType::Vjp => OwnedTokenType::Vjp,
            TokenType::Jvp => OwnedTokenType::Jvp,
            TokenType::Requires => OwnedTokenType::Requires,
            TokenType::Ensures => OwnedTokenType::Ensures,
            TokenType::Invariant => OwnedTokenType::Invariant,
            TokenType::MacroRules => OwnedTokenType::MacroRules,
            TokenType::Topology => OwnedTokenType::Topology,
            TokenType::Memory => OwnedTokenType::Memory,
            TokenType::Ref => OwnedTokenType::Ref,
            TokenType::Verified => OwnedTokenType::Verified,
            TokenType::Pinned => OwnedTokenType::Pinned,
            TokenType::HardwareState => OwnedTokenType::HardwareState,
            TokenType::Identifier(s) => OwnedTokenType::Identifier(s.to_string()),
            TokenType::Number(s) => OwnedTokenType::Number(s.to_string()),
            TokenType::StringLiteral(s) => OwnedTokenType::StringLiteral(s.to_string()),
            TokenType::LeftParen => OwnedTokenType::LeftParen,
            TokenType::RightParen => OwnedTokenType::RightParen,
            TokenType::LeftBrace => OwnedTokenType::LeftBrace,
            TokenType::RightBrace => OwnedTokenType::RightBrace,
            TokenType::LeftBracket => OwnedTokenType::LeftBracket,
            TokenType::RightBracket => OwnedTokenType::RightBracket,
            TokenType::LeftAngle => OwnedTokenType::LeftAngle,
            TokenType::RightAngle => OwnedTokenType::RightAngle,
            TokenType::Colon => OwnedTokenType::Colon,
            TokenType::DoubleColon => OwnedTokenType::DoubleColon,
            TokenType::Semicolon => OwnedTokenType::Semicolon,
            TokenType::Comma => OwnedTokenType::Comma,
            TokenType::Equals => OwnedTokenType::Equals,
            TokenType::PlusEquals => OwnedTokenType::PlusEquals,
            TokenType::Arrow => OwnedTokenType::Arrow,
            TokenType::FatArrow => OwnedTokenType::FatArrow,
            TokenType::Plus => OwnedTokenType::Plus,
            TokenType::Minus => OwnedTokenType::Minus,
            TokenType::Star => OwnedTokenType::Star,
            TokenType::Slash => OwnedTokenType::Slash,
            TokenType::Dot => OwnedTokenType::Dot,
            TokenType::DoubleDot => OwnedTokenType::DoubleDot,
            TokenType::Ampersand => OwnedTokenType::Ampersand,
            TokenType::At => OwnedTokenType::At,
            TokenType::Dollar => OwnedTokenType::Dollar,
            TokenType::EqEq => OwnedTokenType::EqEq,
            TokenType::NotEq => OwnedTokenType::NotEq,
            TokenType::LessEq => OwnedTokenType::LessEq,
            TokenType::GreaterEq => OwnedTokenType::GreaterEq,
            TokenType::AndAnd => OwnedTokenType::AndAnd,
            TokenType::OrOr => OwnedTokenType::OrOr,
            TokenType::Bang => OwnedTokenType::Bang,
            TokenType::Pipe => OwnedTokenType::Pipe,
            TokenType::Eof => OwnedTokenType::Eof,
            TokenType::Unknown(c) => OwnedTokenType::Unknown(c),
            TokenType::Comment(s) => OwnedTokenType::Comment(s.to_string()),
            TokenType::Whitespace(s) => OwnedTokenType::Whitespace(s.to_string()),
        };
        OwnedToken {
            kind,
            line: self.line,
            column: self.column,
            length: self.length,
        }
    }
}

impl std::fmt::Display for OwnedTokenType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OwnedTokenType::Fn => write!(f, "fn"),
            OwnedTokenType::Let => write!(f, "let"),
            OwnedTokenType::Mut => write!(f, "mut"),
            OwnedTokenType::For => write!(f, "for"),
            OwnedTokenType::In => write!(f, "in"),
            OwnedTokenType::If => write!(f, "if"),
            OwnedTokenType::Else => write!(f, "else"),
            OwnedTokenType::Loop => write!(f, "loop"),
            OwnedTokenType::Break => write!(f, "break"),
            OwnedTokenType::Continue => write!(f, "continue"),
            OwnedTokenType::Return => write!(f, "return"),
            OwnedTokenType::Spawn => write!(f, "spawn"),
            OwnedTokenType::On => write!(f, "on"),
            OwnedTokenType::Transfer => write!(f, "transfer"),
            OwnedTokenType::Unroll => write!(f, "unroll"),
            OwnedTokenType::Across => write!(f, "across"),
            OwnedTokenType::Match => write!(f, "match"),
            OwnedTokenType::Struct => write!(f, "struct"),
            OwnedTokenType::Unsafe => write!(f, "unsafe"),
            OwnedTokenType::Safe => write!(f, "safe"),
            OwnedTokenType::Extern => write!(f, "extern"),
            OwnedTokenType::Trait => write!(f, "trait"),
            OwnedTokenType::Impl => write!(f, "impl"),
            OwnedTokenType::Comptime => write!(f, "comptime"),
            OwnedTokenType::Import => write!(f, "import"),
            OwnedTokenType::Assert => write!(f, "assert"),
            OwnedTokenType::Enum => write!(f, "enum"),
            OwnedTokenType::As => write!(f, "as"),
            OwnedTokenType::Grad => write!(f, "grad"),
            OwnedTokenType::Vjp => write!(f, "vjp"),
            OwnedTokenType::Jvp => write!(f, "jvp"),
            OwnedTokenType::Requires => write!(f, "requires"),
            OwnedTokenType::Ensures => write!(f, "ensures"),
            OwnedTokenType::Invariant => write!(f, "invariant"),
            OwnedTokenType::Topology => write!(f, "Topology"),
            OwnedTokenType::Memory => write!(f, "Memory"),
            OwnedTokenType::Ref => write!(f, "Ref"),
            OwnedTokenType::Verified => write!(f, "Verified"),
            OwnedTokenType::Pinned => write!(f, "Pinned"),
            OwnedTokenType::HardwareState => write!(f, "HardwareState"),
            OwnedTokenType::MacroRules => write!(f, "macro_rules"),
            OwnedTokenType::Identifier(s) => write!(f, "{}", s),
            OwnedTokenType::Number(s) => write!(f, "{}", s),
            OwnedTokenType::StringLiteral(s) => write!(f, "\"{}\"", s),
            OwnedTokenType::LeftParen => write!(f, "("),
            OwnedTokenType::RightParen => write!(f, ")"),
            OwnedTokenType::LeftBrace => write!(f, "{{"),
            OwnedTokenType::RightBrace => write!(f, "}}"),
            OwnedTokenType::LeftBracket => write!(f, "["),
            OwnedTokenType::RightBracket => write!(f, "]"),
            OwnedTokenType::LeftAngle => write!(f, "<"),
            OwnedTokenType::RightAngle => write!(f, ">"),
            OwnedTokenType::Colon => write!(f, ":"),
            OwnedTokenType::DoubleColon => write!(f, "::"),
            OwnedTokenType::Semicolon => write!(f, ";"),
            OwnedTokenType::Comma => write!(f, ","),
            OwnedTokenType::Equals => write!(f, "="),
            OwnedTokenType::PlusEquals => write!(f, "+="),
            OwnedTokenType::Arrow => write!(f, "->"),
            OwnedTokenType::FatArrow => write!(f, "=>"),
            OwnedTokenType::Plus => write!(f, "+"),
            OwnedTokenType::Minus => write!(f, "-"),
            OwnedTokenType::Star => write!(f, "*"),
            OwnedTokenType::Slash => write!(f, "/"),
            OwnedTokenType::Dot => write!(f, "."),
            OwnedTokenType::DoubleDot => write!(f, ".."),
            OwnedTokenType::Ampersand => write!(f, "&"),
            OwnedTokenType::At => write!(f, "@"),
            OwnedTokenType::Dollar => write!(f, "$"),
            OwnedTokenType::EqEq => write!(f, "=="),
            OwnedTokenType::NotEq => write!(f, "!="),
            OwnedTokenType::LessEq => write!(f, "<="),
            OwnedTokenType::GreaterEq => write!(f, ">="),
            OwnedTokenType::AndAnd => write!(f, "&&"),
            OwnedTokenType::OrOr => write!(f, "||"),
            OwnedTokenType::Bang => write!(f, "!"),
            OwnedTokenType::Pipe => write!(f, "|"),
            OwnedTokenType::Comment(s) => write!(f, "{}", s),
            OwnedTokenType::Whitespace(s) => write!(f, "{}", s),
            OwnedTokenType::Unknown(c) => write!(f, "{}", c),
            OwnedTokenType::Eof => write!(f, ""),
        }
    }
}
