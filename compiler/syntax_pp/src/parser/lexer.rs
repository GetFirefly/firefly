use std::str::FromStr;

use firefly_diagnostics::*;
use firefly_intern::Symbol;
use firefly_number::{Float, Integer};
use firefly_parser::{Scanner, Source};
use firefly_syntax_erl::LexicalError;

use super::{LexicalToken, Token};

pub struct Lexer<S> {
    scanner: Scanner<S>,
    token: Token,
    token_start: SourceIndex,
    token_end: SourceIndex,
    eof: bool,
    buffer: String,
}
impl<S> Lexer<S>
where
    S: Source,
{
    pub fn new(scanner: Scanner<S>) -> Self {
        let start = scanner.start();
        let mut lexer = Self {
            scanner,
            token: Token::EOF,
            token_start: start,
            token_end: start,
            eof: false,
            buffer: String::new(),
        };
        lexer.advance();
        lexer
    }

    pub fn lex(&mut self) -> Option<<Self as Iterator>::Item> {
        if self.eof && self.token == Token::EOF {
            return None;
        }

        let token = std::mem::replace(&mut self.token, Token::EOF);
        let result = Some(Ok(LexicalToken(
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

    fn read(&mut self) -> char {
        self.scanner.read().1
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

    fn skip_whitespace(&mut self) {
        while self.read().is_whitespace() {
            self.skip();
        }
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

        Token::from_bare_atom(self.slice())
    }

    fn lex_quoted_atom(&mut self) -> Token {
        let c = self.pop();
        debug_assert!(c == '\'');

        self.buffer.clear();

        loop {
            match self.read() {
                '\\' => unimplemented!(),
                '\'' => {
                    self.skip();
                    break;
                }
                c => {
                    self.skip();
                    self.buffer.push(c);
                }
            }
        }

        Token::from_bare_atom(self.buffer.as_str())
    }

    fn lex_string(&mut self) -> Token {
        let c = self.pop();
        debug_assert!(c == '"');

        self.buffer.clear();

        loop {
            match self.read() {
                '\\' => unimplemented!(),
                '"' => {
                    self.skip();
                    break;
                }
                c => {
                    self.skip();
                    self.buffer.push(c);
                }
            }
        }

        Token::StringLiteral(Symbol::intern(&self.buffer))
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
        let negative = c == '-';
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
            return Token::Error(err);
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

    fn lex_float(&mut self) -> Token {
        let c = self.pop();
        debug_assert!(c.is_digit(10));

        while self.read().is_digit(10) {
            self.pop();
        }

        match f64::from_str(self.slice()) {
            Ok(f) => Token::FloatLiteral(Float::new(f)),
            Err(e) => panic!("unhandled float parsing error: {}", &e),
        }
    }

    // Called after consuming a number up to and including the '.'
    #[inline]
    fn lex_float(&mut self, num: String, seen_e: bool) -> Token {
        let mut num = num;

        let mut c = self.pop();
        debug_assert!(c.is_digit(10), "got {}", c);
        num.push(c);

        if let Err(err) = self.lex_digits(10, true, &mut num) {
            return Token::Error(err);
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
                return Token::Error(LexicalError::InvalidFloat {
                    span: self.span(),
                    reason: "expected digits after scientific notation".to_string(),
                });
            }

            if let Err(err) = self.lex_digits(10, false, &mut num) {
                return Token::Error(err);
            }
        }

        self.to_float_literal(num)
    }

    fn to_float_literal(&self, num: String) -> Token {
        let reason = match f64::from_str(&num) {
            Ok(f) => match Float::new(f) {
                Ok(f) => return Token::Float(f),
                Err(FloatError::Nan) => "float cannot be NaN".to_string(),
                Err(FloatError::Infinite) => "float cannot be -Inf or Inf".to_string(),
            },
            Err(e) => e.to_string(),
        };

        Token::Error(LexicalError::InvalidFloat {
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

impl<S> Lexer<S>
where
    S: Source,
{
    fn tokenize(&mut self) -> Token {
        let c = self.read();

        if c == '\0' {
            self.eof = true;
            return Token::EOF;
        }

        if c.is_whitespace() {
            self.skip_whitespace();
        }

        match self.read() {
            '{' => pop!(self, Token::CurlyOpen),
            '}' => pop!(self, Token::CurlyClose),
            '[' => pop!(self, Token::SquareOpen),
            ']' => pop!(self, Token::SquareClose),
            ',' => pop!(self, Token::Comma),
            '.' => pop!(self, Token::Dot),
            '|' => pop!(self, Token::Pipe),
            'a'..='z' | 'A'..='Z' => self.lex_unquoted_atom(),
            '0'..='9' => self.lex_number(),
            '\'' => self.lex_quoted_atom(),
            '"' => self.lex_string(),
            c => unimplemented!("{}", c),
        }
    }
}

impl<S> Iterator for Lexer<S>
where
    S: Source,
{
    type Item = Result<(SourceIndex, Token, SourceIndex), ()>;

    fn next(&mut self) -> Option<Self::Item> {
        self.lex()
    }
}

// Converts the string literal into either a `i64` or arbitrary precision integer, preferring `i64`.
//
// This function panics if the literal is unparseable due to being invalid for the given radix,
// or containing non-ASCII digits.
fn to_integer_literal(literal: &str, radix: u32) -> Token {
    let int = Integer::from_string_radix(literal, radix).unwrap();
    Token::IntegerLiteral(int)
}
