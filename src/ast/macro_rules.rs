use crate::lexer::Token;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenTree {
    Token(Token),
    Group(Vec<TokenTree>),
    Delimited(Delimiter, Vec<TokenTree>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Delimiter {
    Parenthesis, // ()
    Brace,       // {}
    Bracket,     // []
}
