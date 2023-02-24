mod errors;
mod token;

pub use self::errors::LexicalError;
pub use self::token::Token;

use crate::parser::ParserError;

use std::ops::Range;
use std::str::FromStr;

use firefly_diagnostics::*;
use firefly_intern::Symbol;
use firefly_number::{Float, FloatError, Int};
use firefly_parser::{EscapeStm, EscapeStmAction, Scanner, Source};

pub struct Lexer<S> {
    scanner: Scanner<S>,

    /// Escape sequence state machine.
    escape: EscapeStm<SourceIndex>,

    token: Token,
    token_start: SourceIndex,
    token_end: SourceIndex,
    eof: bool,
}
impl<S> Lexer<S>
where
    S: Source,
{
    pub fn new(scanner: Scanner<S>) -> Self {
        let start = scanner.start();
        let mut lexer = Self {
            scanner,
            escape: EscapeStm::new(),
            token: Token::EOF,
            token_start: start,
            token_end: start,
            eof: false,
        };
        lexer.advance();
        lexer
    }

    pub fn lex(&mut self) -> Option<<Self as Iterator>::Item> {
        if self.eof && self.token == Token::EOF {
            return None;
        }

        let token = std::mem::replace(&mut self.token, Token::EOF);
        let result = Some(Ok((
            self.token_start.clone(),
            token,
            self.token_end.clone(),
        )));

        self.advance();

        result
    }

    fn advance(&mut self) {
        self.advance_start();
        self.token = self.tokenize();
    }

    fn advance_start(&mut self) {
        let mut position: SourceIndex;
        loop {
            let (pos, c) = self.scanner.read();
            position = pos;

            if c == '\0' {
                self.eof = true;
                return;
            }

            if c.is_whitespace() {
                self.scanner.advance();
                continue;
            }

            break;
        }

        self.token_start = position;
    }

    fn pop(&mut self) -> char {
        let (pos, c) = self.scanner.pop();
        self.token_end = pos + ByteOffset::from_char_len(c);
        c
    }

    fn peek(&mut self) -> char {
        self.scanner.peek().1
    }

    #[inline]
    fn peek_next(&mut self) -> char {
        let (_, c) = self.scanner.peek_next();
        c
    }

    fn read(&mut self) -> char {
        self.scanner.read().1
    }

    fn index(&mut self) -> SourceIndex {
        self.scanner.read().0
    }

    fn skip(&mut self) {
        self.pop();
    }

    pub fn span(&self) -> SourceSpan {
        SourceSpan::new(self.token_start, self.token_end)
    }

    fn slice(&self) -> &str {
        self.scanner.slice(self.span())
    }

    fn slice_span(&self, span: impl Into<Range<usize>>) -> &str {
        self.scanner.slice(span)
    }

    fn skip_whitespace(&mut self) {
        while self.read().is_whitespace() {
            self.skip();
        }
    }

    fn lex_comment(&mut self) -> Token {
        let mut c = self.read();

        loop {
            if c == '\n' {
                break;
            }

            if c == '\0' {
                self.eof = true;
                break;
            }

            self.skip();
            c = self.read();
        }

        return Token::Comment;
    }

    fn lex_unquoted_atom(&mut self) -> Token {
        let c = self.pop();
        debug_assert!(c.is_ascii_lowercase());

        loop {
            match self.read() {
                '_' => self.skip(),
                '@' => self.skip(),
                '0'..='9' => self.skip(),
                c if c.is_alphanumeric() => self.skip(),
                _ => break,
            }
        }

        Token::from_atom(Symbol::intern(self.slice()))
    }

    #[inline]
    fn lex_string(&mut self) -> Token {
        let quote = self.pop();
        debug_assert!(quote == '"' || quote == '\'');
        let mut buf = None;
        loop {
            match self.read() {
                '\\' => match self.lex_escape_sequence() {
                    Ok(_c) => (),
                    Err(err) => return Token::Err(err),
                },
                '\0' if quote == '"' => {
                    return Token::Err(LexicalError::UnclosedString { span: self.span() });
                }
                '\0' if quote == '\'' => {
                    return Token::Err(LexicalError::UnclosedAtom { span: self.span() });
                }
                c if c == quote => {
                    let span = self.span().shrink_front(ByteOffset(1));

                    self.skip();
                    self.advance_start();
                    if self.read() == quote {
                        self.skip();

                        buf = Some(self.slice_span(span).to_string());
                        continue;
                    }

                    let symbol = if let Some(mut buf) = buf {
                        buf.push_str(self.slice_span(span));
                        Symbol::intern(&buf)
                    } else {
                        Symbol::intern(self.slice_span(span))
                    };

                    let token = Token::StringLiteral(symbol);
                    return token;
                }
                _ => {
                    self.skip();
                    continue;
                }
            }
        }
    }

    #[inline]
    fn lex_escape_sequence(&mut self) -> Result<u64, LexicalError> {
        let start_idx = self.index();

        let c = self.read();
        debug_assert_eq!(c, '\\');

        self.escape.reset();

        let mut byte_idx = 0;

        loop {
            let c = self.read();
            let idx = start_idx + byte_idx;

            let c = if c == '\0' { None } else { Some(c) };
            let res = self.escape.transition(c, idx);

            match res {
                Ok((action, result)) => {
                    if let EscapeStmAction::Next = action {
                        byte_idx += c.map(|c| c.len_utf8()).unwrap_or(0);
                        self.pop();
                    }

                    if let Some(result) = result {
                        return Ok(result.cp);
                    }
                }
                Err(err) => Err(LexicalError::EscapeError { source: err })?,
            }
        }
    }

    #[inline]
    fn lex_digits(
        &mut self,
        radix: u32,
        allow_leading_underscore: bool,
        num: &mut String,
    ) -> Result<(), LexicalError> {
        let mut last_underscore = !allow_leading_underscore;
        let mut c = self.read();
        loop {
            match c {
                c if c.is_digit(radix) => {
                    last_underscore = false;
                    num.push(self.pop());
                }
                '_' if last_underscore => {
                    return Err(LexicalError::UnexpectedCharacter {
                        start: self.span().start(),
                        found: c,
                    });
                }
                '_' if self.peek().is_digit(radix) => {
                    last_underscore = true;
                    self.pop();
                }
                _ => break,
            }
            c = self.read();
        }

        Ok(())
    }

    fn lex_number(&mut self) -> Token {
        let mut num = String::new();
        let mut c;

        // Expect the first character to be either a sign on digit
        c = self.read();
        debug_assert!(c == '-' || c == '+' || c.is_digit(10), "got {}", c);

        // If sign, consume it
        //
        // -10
        // ^
        //
        if c == '-' || c == '+' {
            num.push(self.pop());
        }

        // Consume leading digits
        //
        // -10.0
        //  ^^
        //
        // 10e10
        // ^^
        //
        if let Err(err) = self.lex_digits(10, false, &mut num) {
            return Token::Err(err);
        }

        // If we have a dot with a trailing number, we lex a float.
        // Otherwise we return consumed digits as an integer token.
        //
        // 10.0
        //   ^ lex_float()
        //
        // fn() -> 10 + 10.
        //                ^ return integer token
        //
        c = self.read();
        if c == '.' {
            if self.peek().is_digit(10) {
                // Pushes .
                num.push(self.pop());
                return self.lex_float(num, false);
            }
            return to_integer_literal(&num, 10);
        }

        // Consume exponent marker
        //
        // 10e10
        //   ^ lex_float()
        //
        // 10e-10
        //  ^^ lex_float()
        if c == 'e' || c == 'E' {
            let c2 = self.peek();
            if c2 == '-' || c2 == '+' {
                num.push(self.pop());
                num.push(self.pop());
                return self.lex_float(num, true);
            } else if c2.is_digit(10) {
                num.push(self.pop());
                return self.lex_float(num, true);
            }
        }

        to_integer_literal(&num, 10)
    }

    // Called after consuming a number up to and including the '.'
    #[inline]
    fn lex_float(&mut self, num: String, seen_e: bool) -> Token {
        let mut num = num;

        let mut c = self.pop();
        debug_assert!(c.is_digit(10), "got {}", c);
        num.push(c);

        if let Err(err) = self.lex_digits(10, true, &mut num) {
            return Token::Err(err);
        }

        c = self.read();

        // If we've already seen e|E, then we're done
        if seen_e {
            return self.to_float_literal(num);
        }

        if c == 'E' || c == 'e' {
            num.push(self.pop());
            c = self.read();
            if c == '-' || c == '+' {
                num.push(self.pop());
                c = self.read();
            }

            if !c.is_digit(10) {
                return Token::Err(LexicalError::InvalidFloat {
                    span: self.span(),
                    reason: "expected digits after scientific notation".to_string(),
                });
            }

            if let Err(err) = self.lex_digits(10, false, &mut num) {
                return Token::Err(err);
            }
        }

        self.to_float_literal(num)
    }

    fn to_float_literal(&self, num: String) -> Token {
        let reason = match f64::from_str(&num) {
            Ok(f) => match Float::new(f) {
                Ok(f) => return Token::FloatLiteral(f),
                Err(FloatError::Nan) => "float cannot be NaN".to_string(),
                Err(FloatError::Infinite) => "float cannot be -Inf or Inf".to_string(),
            },
            Err(e) => e.to_string(),
        };

        Token::Err(LexicalError::InvalidFloat {
            span: self.span(),
            reason,
        })
    }
}

macro_rules! pop {
    ($lex:ident) => {{
        $lex.skip();
    }};
    ($lex:ident, $code:expr) => {{
        $lex.skip();
        $code
    }};
}

macro_rules! pop2 {
    ($lex:ident) => {{
        $lex.skip();
        $lex.skip();
    }};
    ($lex:ident, $code:expr) => {{
        $lex.skip();
        $lex.skip();
        $code
    }};
}

impl<S> Lexer<S>
where
    S: Source,
{
    fn tokenize(&mut self) -> Token {
        let c = self.read();

        if c == '%' {
            return self.lex_comment();
        }

        if c == '\0' {
            self.eof = true;
            return Token::EOF;
        }

        if c.is_whitespace() {
            self.skip_whitespace();
        }

        match self.read() {
            '{' => pop!(self, Token::LBrace),
            '}' => pop!(self, Token::RBrace),
            '[' => pop!(self, Token::LBracket),
            ']' => pop!(self, Token::RBracket),
            '#' => pop!(self, Token::Pound),
            '=' => match self.peek() {
                '>' => pop2!(self, Token::RightArrow),
                _ => Token::Err(LexicalError::UnexpectedCharacter {
                    start: self.span().start(),
                    found: self.peek_next(),
                }),
            },
            '|' => pop!(self, Token::Bar),
            ',' => pop!(self, Token::Comma),
            '.' => pop!(self, Token::Dot),
            'a'..='z' | 'A'..='Z' => self.lex_unquoted_atom(),
            '0'..='9' => self.lex_number(),
            '"' => self.lex_string(),
            '\'' => match self.lex_string() {
                Token::StringLiteral(s) => Token::from_atom(s),
                other => other,
            },
            c => unimplemented!("{}", c),
        }
    }
}

impl<S> Iterator for Lexer<S>
where
    S: Source,
{
    type Item = Result<(SourceIndex, Token, SourceIndex), ParserError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut res = self.lex();
        loop {
            match res {
                Some(Ok((_, Token::Comment, _))) => {
                    res = self.lex();
                }
                _ => break,
            }
        }
        res.map(|result| result.map_err(|err| err.into()))
    }
}

// Converts the string literal into either a `i64` or arbitrary precision integer, preferring `i64`.
//
// This function panics if the literal is unparseable due to being invalid for the given radix,
// or containing non-ASCII digits.
fn to_integer_literal(literal: &str, radix: u32) -> Token {
    let int = Int::from_string_radix(literal, radix).unwrap();
    Token::IntegerLiteral(int)
}
