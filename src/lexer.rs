#[derive(Debug, PartialEq, Clone)]
pub enum TokenType {
    // Keywords
    Fn,
    Let,
    Mut,
    For,
    In,
    If,
    Else,
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
    StringLiteral(String),

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
    Comment(String),
    Whitespace(String),
}

impl std::fmt::Display for TokenType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenType::Fn => write!(f, "fn"),
            TokenType::Let => write!(f, "let"),
            TokenType::Mut => write!(f, "mut"),
            TokenType::For => write!(f, "for"),
            TokenType::In => write!(f, "in"),
            TokenType::If => write!(f, "if"),
            TokenType::Else => write!(f, "else"),
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

            TokenType::Topology => write!(f, "Topology"),
            TokenType::Memory => write!(f, "Memory"),
            TokenType::Ref => write!(f, "Ref"),
            TokenType::Verified => write!(f, "Verified"),
            TokenType::Pinned => write!(f, "Pinned"),
            TokenType::HardwareState => write!(f, "HardwareState"),

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
            TokenType::Plus => write!(f, "+"),
            TokenType::Minus => write!(f, "-"),
            TokenType::Star => write!(f, "*"),
            TokenType::Slash => write!(f, "/"),
            TokenType::Dot => write!(f, "."),
            TokenType::DoubleDot => write!(f, ".."),
            TokenType::Ampersand => write!(f, "&"),

            TokenType::EqEq => write!(f, "=="),
            TokenType::NotEq => write!(f, "!="),
            TokenType::LessEq => write!(f, "<="),
            TokenType::GreaterEq => write!(f, ">="),
            TokenType::AndAnd => write!(f, "&&"),
            TokenType::OrOr => write!(f, "||"),
            TokenType::Bang => write!(f, "!"),

            TokenType::Comment(s) => write!(f, "{}", s),
            TokenType::Whitespace(s) => write!(f, "{}", s),
            TokenType::Unknown(c) => write!(f, "{}", c),
            TokenType::Eof => write!(f, ""),
            TokenType::Enum => write!(f, "enum"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenType,
    pub line: usize,
    pub column: usize,
    pub length: usize,
}

pub struct Lexer<'a> {
    source: std::iter::Peekable<std::str::Chars<'a>>,
    line: usize,
    column: usize,
    pub preserve_comments: bool,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Self {
            source: source.chars().peekable(),
            line: 1,
            column: 1,
            preserve_comments: false,
        }
    }

    pub fn new_with_comments(source: &'a str) -> Self {
        Self {
            source: source.chars().peekable(),
            line: 1,
            column: 1,
            preserve_comments: true,
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
            "if" => TokenType::If,
            "else" => TokenType::Else,
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
            _ => TokenType::Identifier(text.clone()),
        };

        Token {
            kind,
            line: self.line,
            column: start_col,
            length: text.len(),
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
            kind: TokenType::Number(text.clone()),
            line: self.line,
            column: start_col,
            length: text.len(),
        }
    }

    pub fn next_token(&mut self) -> Token {
        if !self.preserve_comments {
            self.skip_whitespace();
        }

        let start_col = self.column;
        let c = match self.peek() {
            Some(&c) => c,
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
                let mut ws = String::new();
                while let Some(&c) = self.peek() {
                    if c.is_whitespace() {
                        ws.push(self.advance().unwrap());
                    } else {
                        break;
                    }
                }
                return Token {
                    kind: TokenType::Whitespace(ws.clone()),
                    line: self.line,
                    column: start_col,
                    length: ws.len(),
                };
            }

            if c == '/' {
                let mut temp = self.source.clone();
                temp.next();
                if temp.peek() == Some(&'/') {
                    let mut comment = String::new();
                    comment.push(self.advance().unwrap()); // '/'
                    comment.push(self.advance().unwrap()); // '/'
                    while let Some(&next_c) = self.peek() {
                        if next_c == '\n' {
                            break;
                        }
                        comment.push(self.advance().unwrap());
                    }
                    return Token {
                        kind: TokenType::Comment(comment.clone()),
                        line: self.line,
                        column: start_col,
                        length: comment.len(),
                    };
                }
            }
        }

        let c = self.advance().unwrap();

        if c.is_alphabetic() || c == '_' {
            return self.identifier_or_keyword(c, start_col);
        }

        if c.is_ascii_digit() {
            return self.number(c, start_col);
        }

        if c == '"' {
            let mut text = String::new();
            while let Some(&next_c) = self.peek() {
                if next_c == '"' {
                    self.advance();
                    break;
                }
                let mut char_to_push = self.advance().unwrap();
                if char_to_push == '\\' {
                    if let Some(&esc_c) = self.peek() {
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
                text.push(char_to_push);
            }
            return Token {
                kind: TokenType::StringLiteral(text.clone()),
                line: self.line,
                column: start_col,
                length: text.len() + 2, // including the quotes
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
            length: self.column - start_col,
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
