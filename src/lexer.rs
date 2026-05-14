#[derive(Debug, PartialEq, Clone)]
pub enum TokenType {
    // Keywords
    Fn,
    Let,
    Mut,
    For,
    In,
    Return,
    Spawn,
    On,
    Transfer,
    Unroll,
    Across,
    Match,
    Struct,
    Unsafe,

    // Types & Topology
    Topology,
    Memory,
    Ref,
    Verified,
    Pinned,
    HardwareState,

    // Identifiers & Literals
    Identifier(String),
    Number(String),

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
    Plus,
    Minus,
    Star,
    Slash,
    Dot,
    DoubleDot,
    Ampersand,

    // Logical & Relational
    EqEq,
    NotEq,
    LessEq,
    GreaterEq,
    AndAnd,
    OrOr,
    Bang,

    // Special
    Eof,
    Unknown(char),
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenType,
    pub line: usize,
    pub column: usize,
}

pub struct Lexer<'a> {
    source: std::iter::Peekable<std::str::Chars<'a>>,
    line: usize,
    column: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Self {
            source: source.chars().peekable(),
            line: 1,
            column: 1,
        }
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.source.next()?;
        if c == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(c)
    }

    fn peek(&mut self) -> Option<&char> {
        self.source.peek()
    }

    fn skip_whitespace(&mut self) {
        while let Some(&c) = self.peek() {
            if c.is_whitespace() {
                self.advance();
            } else if c == '/' {
                // Peek ahead to check for comments
                let mut temp = self.source.clone();
                temp.next(); // consume '/'
                if temp.peek() == Some(&'/') {
                    // Line comment
                    while let Some(c) = self.advance() {
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

    fn identifier_or_keyword(&mut self, start_char: char, start_col: usize) -> Token {
        let mut text = String::new();
        text.push(start_char);

        while let Some(&c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                text.push(self.advance().unwrap());
            } else {
                break;
            }
        }

        let kind = match text.as_str() {
            "fn" => TokenType::Fn,
            "let" => TokenType::Let,
            "mut" => TokenType::Mut,
            "for" => TokenType::For,
            "in" => TokenType::In,
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
            _ => TokenType::Identifier(text),
        };

        Token {
            kind,
            line: self.line,
            column: start_col,
        }
    }

    fn number(&mut self, start_char: char, start_col: usize) -> Token {
        let mut text = String::new();
        text.push(start_char);

        while let Some(&c) = self.peek() {
            if c.is_ascii_digit() {
                text.push(self.advance().unwrap());
            } else if c == '.' {
                let mut temp = self.source.clone();
                temp.next();
                if temp.peek() == Some(&'.') {
                    break;
                }
                text.push(self.advance().unwrap());
            } else {
                break;
            }
        }

        Token {
            kind: TokenType::Number(text),
            line: self.line,
            column: start_col,
        }
    }

    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace();

        let start_col = self.column;
        let c = match self.advance() {
            Some(c) => c,
            None => {
                return Token {
                    kind: TokenType::Eof,
                    line: self.line,
                    column: self.column,
                }
            }
        };

        if c.is_alphabetic() || c == '_' {
            return self.identifier_or_keyword(c, start_col);
        }

        if c.is_ascii_digit() {
            return self.number(c, start_col);
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
                if self.peek() == Some(&'=') {
                    self.advance();
                    TokenType::PlusEquals
                } else {
                    TokenType::Plus
                }
            }
            '*' => TokenType::Star,
            '/' => TokenType::Slash,
            '=' => {
                if self.peek() == Some(&'=') {
                    self.advance();
                    TokenType::EqEq
                } else {
                    TokenType::Equals
                }
            }
            '!' => {
                if self.peek() == Some(&'=') {
                    self.advance();
                    TokenType::NotEq
                } else {
                    TokenType::Bang
                }
            }
            '<' => {
                if self.peek() == Some(&'=') {
                    self.advance();
                    TokenType::LessEq
                } else {
                    TokenType::LeftAngle
                }
            }
            '>' => {
                if self.peek() == Some(&'=') {
                    self.advance();
                    TokenType::GreaterEq
                } else {
                    TokenType::RightAngle
                }
            }
            '&' => {
                if let Some(&'&') = self.peek() {
                    self.advance();
                    TokenType::AndAnd
                } else {
                    TokenType::Ampersand
                }
            }
            '|' => {
                if self.peek() == Some(&'|') {
                    self.advance();
                    TokenType::OrOr
                } else {
                    TokenType::Unknown('|')
                }
            }
            '.' => {
                if self.peek() == Some(&'.') {
                    self.advance();
                    TokenType::DoubleDot
                } else {
                    TokenType::Dot
                }
            }
            '-' => {
                if self.peek() == Some(&'>') {
                    self.advance();
                    TokenType::Arrow
                } else {
                    TokenType::Minus
                }
            }
            ':' => {
                if self.peek() == Some(&':') {
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
        }
    }

    pub fn tokenize(&mut self) -> Vec<Token> {
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
