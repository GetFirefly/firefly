use std::path::Path;

use super::{LexicalToken, Position, Result, Tokenizer};

/// Lexer.
///
/// `Lexer` is similar to the `Tokenizer`.
/// But unlike `Tokenizer`, `Lexer` returns only lexical tokens (i.e., hidden tokens are discarded).
#[derive(Debug)]
pub struct Lexer<T>(Tokenizer<T>);
impl<T> Lexer<T>
where
    T: AsRef<str>,
{
    /// Makes a new `Lexer` instance which tokenize the Erlang source code text.
    pub fn new(text: T) -> Self {
        Lexer(Tokenizer::new(text))
    }

    /// Sets the file path of the succeeding tokens.
    pub fn set_filepath<P: AsRef<Path>>(&mut self, filepath: P) {
        self.0.set_filepath(filepath);
    }

    /// Returns the input text.
    pub fn text(&self) -> &str {
        self.0.text()
    }

    /// Finishes tokenization and returns the target text.
    pub fn finish(self) -> T {
        self.0.finish()
    }

    /// Returns the cursor position from which this lexer will start to scan the next token.
    ///
    /// # Examples
    ///
    /// ```
    /// use erl_tokenize::Lexer;
    ///
    /// let src = r#"io:format(
    ///   "Hello")."#;
    ///
    /// let mut lexer = Lexer::new(src);
    /// assert_eq!(lexer.next_position().offset(), 0);
    ///
    /// assert_eq!(lexer.next().unwrap().map(|t| t.text().to_owned()).unwrap(), "io");
    /// assert_eq!(lexer.next_position().offset(), 2);
    /// lexer.next(); // ':'
    /// lexer.next(); // 'format'
    /// lexer.next(); // '('
    /// assert_eq!(lexer.next().unwrap().map(|t| t.text().to_owned()).unwrap(), r#""Hello""#);
    /// assert_eq!(lexer.next_position().offset(), 20);
    /// assert_eq!(lexer.next_position().line(), 2);
    /// assert_eq!(lexer.next_position().column(), 8);
    /// ```
    pub fn next_position(&self) -> Position {
        self.0.next_position()
    }
}
impl<T> Iterator for Lexer<T>
where
    T: AsRef<str>,
{
    type Item = Result<LexicalToken>;
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(token) = self.0.next() {
            match token {
                Err(e) => return Some(Err(e)),
                Ok(token) => {
                    if let Ok(token) = token.into_lexical_token() {
                        return Some(Ok(token));
                    }
                }
            }
        }
        None
    }
}
