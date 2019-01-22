use std::error::Error;
use std::str::FromStr;

use rug::Integer;

use liblumen_diagnostics::{ByteIndex, ByteOffset, ByteSpan};

use super::errors::LexicalError;
use super::scanner::Scanner;
use super::source::Source;
use super::token::*;
use super::{Lexed, Symbol};

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

macro_rules! pop3 {
    ($lex:ident) => {{
        $lex.skip();
        $lex.skip();
        $lex.skip()
    }};
    ($lex:ident, $code:expr) => {{
        $lex.skip();
        $lex.skip();
        $lex.skip();
        $code
    }};
}

/// The lexer that is used to perform lexical analysis on the Erlang grammar. The lexer implements
/// the `Iterator` trait, so in order to retrieve the tokens, you simply have to iterate over it.
///
/// # Errors
///
/// Because the lexer is implemented as an iterator over tokens, this means that you can continue
/// to get tokens even if a lexical error occurs. The lexer will attempt to recover from an error
/// by injecting tokens it expects.
///
/// If an error is unrecoverable, the lexer will continue to produce tokens, but there is no
/// guarantee that parsing them will produce meaningful results, it is primarily to assist in
/// gathering as many errors as possible.
pub struct Lexer<S> {
    /// The scanner produces a sequence of chars + location, and can be controlled
    /// The location produces is a ByteIndex
    scanner: Scanner<S>,

    /// The most recent token to be lexed.
    /// At the start and end, this should be Token::EOF
    token: Token,

    /// The position in the input where the current token starts
    /// At the start this will be the byte index of the beginning of the input
    token_start: ByteIndex,

    /// The position in the input where the current token ends
    /// At the start this will be the byte index of the beginning of the input
    token_end: ByteIndex,

    /// When we have reached true EOF, this gets set to true, and the only token
    /// produced after that point is Token::EOF, or None, depending on how you are
    /// consuming the lexer
    eof: bool,
}

impl<S> Lexer<S>
where
    S: Source,
{
    /// Produces an instance of the lexer with the lexical analysis to be performed on the `input`
    /// string. Note that no lexical analysis occurs until the lexer has been iterated over.
    pub fn new(scanner: Scanner<S>) -> Self {
        let start = scanner.start();
        let mut lexer = Lexer {
            scanner,
            token: Token::EOF,
            token_start: start + ByteOffset(0),
            token_end: start + ByteOffset(0),
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
        let result = if let Token::Error(err) = token {
            Some(Err(err))
        } else {
            Some(Ok(LexicalToken(
                self.token_start.clone(),
                token,
                self.token_end.clone(),
            )))
        };

        self.advance();

        result
    }

    fn advance(&mut self) {
        self.advance_start();
        self.token = self.tokenize();
    }

    #[inline]
    fn advance_start(&mut self) {
        let mut position: ByteIndex;
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

    #[inline]
    fn pop(&mut self) -> char {
        let (pos, c) = self.scanner.pop();
        self.token_end = pos + ByteOffset::from_char_utf8(c);
        c
    }

    #[inline]
    fn peek(&mut self) -> char {
        let (_, c) = self.scanner.peek();
        c
    }

    #[inline]
    fn peek_next(&mut self) -> char {
        let (_, c) = self.scanner.peek_next();
        c
    }

    #[inline]
    fn read(&mut self) -> char {
        let (_, c) = self.scanner.read();
        c
    }

    #[inline]
    fn skip(&mut self) {
        self.pop();
    }

    /// Get the span for the current token in `Source`.
    #[inline]
    pub fn span(&self) -> ByteSpan {
        ByteSpan::new(self.token_start, self.token_end)
    }

    /// Get a string slice of the current token.
    #[inline]
    fn slice(&self) -> &str {
        self.scanner.slice(self.span())
    }

    #[inline]
    fn slice_span(&self, span: ByteSpan) -> &str {
        self.scanner.slice(span)
    }

    #[inline]
    fn skip_whitespace(&mut self) {
        let mut c: char;
        loop {
            c = self.read();

            if !c.is_whitespace() {
                break;
            }

            self.skip();
        }
    }

    fn tokenize(&mut self) -> Token {
        let c = self.read();

        if c == '%' {
            self.skip();
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
            ',' => pop!(self, Token::Comma),
            ';' => pop!(self, Token::Semicolon),
            '_' => self.lex_identifier(),
            '0'...'9' => self.lex_number(),
            'a'...'z' => self.lex_bare_atom(),
            'A'...'Z' => self.lex_identifier(),
            '#' => pop!(self, Token::Pound),
            '*' => pop!(self, Token::Star),
            '!' => pop!(self, Token::Bang),
            '[' => pop!(self, Token::LBracket),
            ']' => pop!(self, Token::RBracket),
            '(' => pop!(self, Token::LParen),
            ')' => pop!(self, Token::RParen),
            '{' => pop!(self, Token::LBrace),
            '}' => pop!(self, Token::RBrace),
            '?' => match self.peek() {
                '?' => pop2!(self, Token::DoubleQuestion),
                _ => pop!(self, Token::Question),
            },
            '-' => match self.peek() {
                '-' => pop2!(self, Token::MinusMinus),
                '>' => pop2!(self, Token::RightStab),
                n if n.is_digit(10) => self.lex_number(),
                _ => pop!(self, Token::Minus),
            },
            '$' => {
                self.skip();
                let c = self.pop();
                if c == '\\' {
                    return match self.lex_escape_sequence() {
                        Ok(Token::Char(c)) => Token::Char(c),
                        Ok(Token::Integer(i)) => match std::char::from_u32(i as u32) {
                            Some(c) => Token::Char(c),
                            None => Token::Error(LexicalError::InvalidEscape {
                                span: self.span(),
                                reason: format!("the integer value '{}' is not a valid char", i),
                            }),
                        },
                        Ok(Token::BigInteger(i)) => Token::Error(LexicalError::InvalidEscape {
                            span: self.span(),
                            reason: format!("the integer value '{}' is not a valid char", i),
                        }),
                        Ok(_) => panic!("internal error: unhandled escape sequence in lexer"),
                        Err(e) => Token::Error(e),
                    };
                }
                Token::Char(c)
            }
            '"' => self.lex_string(),
            '\'' => match self.lex_string() {
                Token::String(s) => Token::Atom(s),
                other => other,
            },
            ':' => match self.peek() {
                '=' => pop2!(self, Token::ColonEqual),
                ':' => pop2!(self, Token::ColonColon),
                _ => pop!(self, Token::Colon),
            },
            '+' => {
                if self.peek() == '+' {
                    pop2!(self, Token::PlusPlus)
                } else {
                    pop!(self, Token::Plus)
                }
            }
            '/' => {
                if self.peek() == '=' {
                    pop2!(self, Token::IsNotEqual)
                } else {
                    pop!(self, Token::Slash)
                }
            }
            '=' => match self.peek() {
                '=' => pop2!(self, Token::IsEqual),
                '>' => pop2!(self, Token::RightArrow),
                '/' => {
                    if self.peek_next() == '=' {
                        pop3!(self, Token::IsExactlyNotEqual)
                    } else {
                        Token::Error(LexicalError::UnexpectedCharacter {
                            start: self.span().start(),
                            found: self.peek_next(),
                        })
                    }
                }
                _ => pop!(self, Token::Equals),
            },
            '<' => match self.peek() {
                '<' => pop2!(self, Token::BinaryStart),
                '-' => pop2!(self, Token::LeftStab),
                '=' => pop2!(self, Token::LeftArrow),
                _ => pop!(self, Token::IsLessThan),
            },
            '>' => match self.peek() {
                '>' => pop2!(self, Token::BinaryEnd),
                '=' => pop2!(self, Token::IsGreaterThanOrEqual),
                _ => pop!(self, Token::IsGreaterThan),
            },
            '|' => {
                if self.peek() == '|' {
                    pop2!(self, Token::BarBar)
                } else {
                    pop!(self, Token::Bar)
                }
            }
            '.' => {
                if self.peek() == '.' {
                    if self.peek_next() == '.' {
                        pop3!(self, Token::DotDotDot)
                    } else {
                        pop2!(self, Token::DotDot)
                    }
                } else {
                    pop!(self, Token::Dot)
                }
            }
            '\\' => {
                // Allow escaping newlines
                let c = self.peek();
                if c == '\n' {
                    pop2!(self);
                    return self.tokenize();
                }
                return match self.lex_escape_sequence() {
                    Ok(t) => t,
                    Err(e) => Token::Error(e),
                };
            }
            c => Token::Error(LexicalError::UnexpectedCharacter {
                start: self.span().start(),
                found: c,
            }),
        }
    }

    fn lex_comment(&mut self) -> Token {
        let mut c = self.read();
        // If there is another '%', then this is a regular comment line
        if c == '%' {
            self.skip();

            loop {
                c = self.read();

                if c == '\n' {
                    break;
                }

                if c == '\0' {
                    self.eof = true;
                    break;
                }

                self.skip();
            }

            return Token::Comment;
        }

        // If no '%', then we should check for an Edoc tag, first skip all whitespace and advance
        // the token start
        self.skip_whitespace();

        // See if this is an Edoc tag
        c = self.read();
        if c == '@' {
            if self.peek().is_ascii_alphabetic() {
                self.skip();

                // Get the tag identifier
                self.lex_identifier();
                // Skip any leading whitespace in the value
                self.skip_whitespace();
                // Get value
                loop {
                    c = self.read();

                    if c == '\n' {
                        break;
                    }

                    if c == '\0' {
                        self.eof = true;
                        break;
                    }

                    self.skip();
                }
                return Token::Edoc;
            }
        }

        // If we reach here, its a normal comment
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

    #[inline]
    fn lex_escape_sequence(&mut self) -> Result<Token, LexicalError> {
        let mut c = self.pop();
        debug_assert_eq!(c, '\\');

        match self.pop() {
            'n' => Ok(Token::Char('\n')),
            'r' => Ok(Token::Char('\r')),
            't' => Ok(Token::Char('\t')),
            'b' => Ok(Token::Char('\x08')),
            // TODO: Figure out why Erlang lexes this as an escape: 'd' => Some('\d'),
            'e' => Ok(Token::Char('\x1B')),
            'f' => Ok(Token::Char('\x0C')),
            's' => Ok(Token::Char(' ')),
            'v' => Ok(Token::Char('\x0B')),
            '\'' => Ok(Token::Char('\'')),
            '"' => Ok(Token::Char('"')),
            '\\' => Ok(Token::Char('\\')),
            // Possible octal escape
            '0' => {
                c = self.read();
                if c.is_digit(8) {
                    let mut num = String::new();
                    while self.read().is_digit(8) {
                        num.push(self.pop());
                    }
                    return Ok(to_integer_literal(&num, 8));
                } else {
                    Ok(Token::Char('\0'))
                }
            }
            // Hex escape
            'x' => {
                c = self.read();
                // \xXY
                if c.is_digit(16) {
                    let mut num = String::new();
                    num.push(self.pop());
                    c = self.read();
                    if c.is_digit(16) {
                        num.push(self.pop());
                        return Ok(to_integer_literal(&num, 16));
                    } else {
                        return Err(LexicalError::InvalidEscape {
                            span: self.span(),
                            reason: "invalid hex escape, expected hex digit".to_string(),
                        });
                    }
                } else if c == '{' {
                    self.skip();
                    let mut num = String::new();
                    while self.read().is_digit(16) {
                        num.push(self.pop());
                    }
                    if self.read() == '}' {
                        self.skip();
                        if num.len() == 0 {
                            return Err(LexicalError::InvalidEscape {
                                span: self.span(),
                                reason: "invalid hex escape, must be at least one digit"
                                    .to_string(),
                            });
                        }
                        return Ok(to_integer_literal(&num, 16));
                    } else {
                        Err(LexicalError::InvalidEscape {
                            span: self.span(),
                            reason: "invalid hex escape, no closing '}'".to_string(),
                        })
                    }
                } else {
                    Err(LexicalError::InvalidEscape {
                        span: self.span(),
                        reason: "invalid hex escape, expected hex digit or '{'".to_string(),
                    })
                }
            }
            _ => Err(LexicalError::InvalidEscape {
                span: self.span(),
                reason: "invalid escape, unrecognized sequence".to_string(),
            }),
        }
    }

    #[inline]
    fn lex_string(&mut self) -> Token {
        let quote = self.pop();
        debug_assert!(quote == '"' || quote == '\'');
        loop {
            match self.read() {
                '\\' => match self.lex_escape_sequence() {
                    Ok(_c) => (),
                    Err(err) => return Token::Error(err),
                },
                '\0' if quote == '"' => {
                    return Token::Error(LexicalError::UnclosedString { span: self.span() });
                }
                '\0' if quote == '\'' => {
                    return Token::Error(LexicalError::UnclosedAtom { span: self.span() });
                }
                c if c == quote => {
                    let symbol =
                        Symbol::intern(self.slice_span(self.span().shrink_front(ByteOffset(1))));
                    let token = Token::String(symbol);
                    self.skip();
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
    fn lex_identifier(&mut self) -> Token {
        let c = self.pop();
        debug_assert!(c == '_' || c.is_ascii_alphabetic());

        loop {
            match self.read() {
                '_' => self.skip(),
                '@' => self.skip(),
                '0'...'9' => self.skip(),
                c if c.is_alphabetic() => self.skip(),
                _ => break,
            }
        }
        Token::Ident(Symbol::intern(self.slice()))
    }

    #[inline]
    fn lex_bare_atom(&mut self) -> Token {
        let c = self.pop();
        debug_assert!(c.is_ascii_lowercase());

        loop {
            match self.read() {
                '_' => self.skip(),
                '@' => self.skip(),
                '0'...'9' => self.skip(),
                c if c.is_alphabetic() => self.skip(),
                _ => break,
            }
        }
        Token::from_bare_atom(self.slice())
    }

    #[inline]
    fn lex_number(&mut self) -> Token {
        let mut num = String::new();
        let mut c = self.pop();
        debug_assert!(c == '-' || c == '+' || c.is_digit(10));

        let negative = c == '-';
        num.push(c);
        // Parse leading digits
        while self.read().is_digit(10) {
            num.push(self.pop());
        }
        c = self.read();
        if c == '.' {
            if self.peek().is_digit(10) {
                // Pushes .
                num.push(self.pop());
                return self.lex_float(num, false);
            }
            return to_integer_literal(&num, 10);
        }
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
        // If followed by '#', the leading digits were the radix
        if c == '#' {
            self.skip();
            // Parse in the given radix
            let radix = match num[1..].parse::<u32>() {
                Ok(r) => r,
                Err(e) => {
                    return Token::Error(LexicalError::InvalidRadix {
                        span: self.span(),
                        reason: e.description().to_string(),
                    });
                }
            };
            if radix >= 2 && radix <= 32 {
                c = self.read();
                if c.is_digit(radix) {
                    let mut num = String::new();
                    if negative {
                        num.push('-');
                    }
                    num.push(self.pop());
                    while self.read().is_digit(radix) {
                        num.push(self.pop());
                    }
                    return to_integer_literal(&num, radix);
                } else {
                    Token::Error(LexicalError::UnexpectedCharacter {
                        start: self.span().start(),
                        found: c,
                    })
                }
            } else {
                Token::Error(LexicalError::InvalidRadix {
                    span: self.span(),
                    reason: "invalid radix (must be in 2..32)".to_string(),
                })
            }
        } else {
            to_integer_literal(&num, 10)
        }
    }

    // Called after consuming a number up to and including the '.'
    #[inline]
    fn lex_float(&mut self, num: String, seen_e: bool) -> Token {
        let mut num = num;
        let mut c = self.pop();
        debug_assert!(c.is_digit(10));

        num.push(c);

        while self.read().is_digit(10) {
            num.push(self.pop());
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
            if c.is_digit(10) {
                while self.read().is_digit(10) {
                    num.push(self.pop());
                }

                return self.to_float_literal(num);
            }
            return Token::Error(LexicalError::InvalidFloat {
                span: self.span(),
                reason: "expected digits after scientific notation".to_string(),
            });
        }
        self.to_float_literal(num)
    }

    fn to_float_literal(&self, num: String) -> Token {
        match f64::from_str(&num) {
            Ok(f) => Token::Float(f),
            Err(e) => Token::Error(LexicalError::InvalidFloat {
                span: self.span(),
                reason: e.description().to_string(),
            }),
        }
    }
}

impl<S> Iterator for Lexer<S>
where
    S: Source,
{
    type Item = Lexed;

    fn next(&mut self) -> Option<Self::Item> {
        self.lex()
    }
}

// Converts the string literal into either a `i64` or arbitrary precision integer, preferring `i64`.
//
// This function panics if the literal is unparseable due to being invalid for the given radix,
// or containing non-ASCII digits.
fn to_integer_literal(literal: &str, radix: u32) -> Token {
    if let Ok(i) = i64::from_str_radix(literal, radix) {
        return Token::Integer(i);
    }
    let bi = Integer::from_str_radix(literal, radix as i32).unwrap();
    Token::BigInteger(bi)
}

#[cfg(test)]
mod test {
    use std::borrow::Cow;
    use std::sync::Arc;

    use liblumen_diagnostics::{ByteIndex, ByteSpan, FileMap, FileName};
    use pretty_assertions::assert_eq;

    use crate::lexer::*;

    macro_rules! symbol {
        ($sym:expr) => {
            Symbol::intern($sym)
        };
    }

    macro_rules! assert_lex(
        ($input:expr, $expected:expr) => ({
            let filemap = FileMap::new(FileName::Virtual(Cow::Borrowed("nofile")), $input);
            let source = FileMapSource::new(Arc::new(filemap));
            let scanner = Scanner::new(source);
            let lexer = Lexer::new(scanner);
            let results = lexer.map(|result| {
                match result {
                    Ok(LexicalToken(start, token, end)) => {
                        Ok((start.to_usize(), token, end.to_usize()))
                    }
                    Err(err) =>  {
                        Err(err)
                    }
                }
            }).collect::<Vec<_>>();
            assert_eq!(results, $expected);
        })
    );

    #[test]
    fn lex_symbols() {
        assert_lex!(":", vec![Ok((1, Token::Colon, 2))]);
        assert_lex!(",", vec![Ok((1, Token::Comma, 2))]);
        assert_lex!("=", vec![Ok((1, Token::Equals, 2))]);
    }

    #[test]
    fn lex_comment() {
        assert_lex!("% this is a comment", vec![Ok((1, Token::Comment, 20))]);
        assert_lex!("% @author Paul", vec![Ok((1, Token::Edoc, 15))]);
    }

    #[test]
    fn lex_float_literal() {
        // With leading 0
        assert_lex!("0.0", vec![Ok((1, Token::Float(0.0), 4))]);
        assert_lex!("000051.0", vec![Ok((1, Token::Float(51.0), 9))]);
        assert_lex!("05162.0", vec![Ok((1, Token::Float(5162.0), 8))]);
        assert_lex!("099.0", vec![Ok((1, Token::Float(99.0), 6))]);
        assert_lex!("04624.51235", vec![Ok((1, Token::Float(4624.51235), 12))]);
        assert_lex!("0.987", vec![Ok((1, Token::Float(0.987), 6))]);
        assert_lex!("0.55e10", vec![Ok((1, Token::Float(0.55e10), 8))]);
        assert_lex!("612.0e61", vec![Ok((1, Token::Float(612e61), 9))]);
        assert_lex!("0.0e-1", vec![Ok((1, Token::Float(0e-1), 7))]);
        assert_lex!("41.0e+9", vec![Ok((1, Token::Float(41e+9), 8))]);

        // Without leading 0
        assert_lex!("5162.0", vec![Ok((1, Token::Float(5162.0), 7))]);
        assert_lex!("99.0", vec![Ok((1, Token::Float(99.0), 5))]);
        assert_lex!("4624.51235", vec![Ok((1, Token::Float(4624.51235), 11))]);
        assert_lex!("612.0e61", vec![Ok((1, Token::Float(612e61), 9))]);
        assert_lex!("41.0e+9", vec![Ok((1, Token::Float(41e+9), 8))]);

        // With leading negative sign
        assert_lex!("-700.5", vec![Ok((1, Token::Float(-700.5), 7))]);
        assert_lex!("-9.0e2", vec![Ok((1, Token::Float(-9.0e2), 7))]);
        assert_lex!("-0.5e1", vec![Ok((1, Token::Float(-0.5e1), 7))]);
        assert_lex!("-0.0", vec![Ok((1, Token::Float(-0.0), 5))]);
    }

    #[test]
    fn lex_identifier_or_atom() {
        assert_lex!(
            "_identifier",
            vec![Ok((1, Token::Ident(symbol!("_identifier")), 12))]
        );
        assert_lex!(
            "_Identifier",
            vec![Ok((1, Token::Ident(symbol!("_Identifier")), 12))]
        );
        assert_lex!(
            "identifier",
            vec![Ok((1, Token::Atom(symbol!("identifier")), 11))]
        );
        assert_lex!(
            "Identifier",
            vec![Ok((1, Token::Ident(symbol!("Identifier")), 11))]
        );
        assert_lex!("z0123", vec![Ok((1, Token::Atom(symbol!("z0123")), 6))]);
        assert_lex!(
            "i_d@e_t0123",
            vec![Ok((1, Token::Atom(symbol!("i_d@e_t0123")), 12))]
        );
    }

    #[test]
    fn lex_integer_literal() {
        // Decimal
        assert_lex!("1", vec![Ok((1, Token::Integer(1), 2))]);
        assert_lex!("9624", vec![Ok((1, Token::Integer(9624), 5))]);
        assert_lex!("-1", vec![Ok((1, Token::Integer(-1), 3))]);
        assert_lex!("-9624", vec![Ok((1, Token::Integer(-9624), 6))]);

        // Hexadecimal
        assert_lex!(r#"\x00"#, vec![Ok((1, Token::Integer(0x0), 5))]);
        assert_lex!(r#"\x{1234FF}"#, vec![Ok((1, Token::Integer(0x1234FF), 11))]);
        assert_lex!("-16#0", vec![Ok((1, Token::Integer(0x0), 6))]);
        assert_lex!("-16#1234FF", vec![Ok((1, Token::Integer(-0x1234FF), 11))]);

        // Octal
        assert_lex!(r#"\00"#, vec![Ok((1, Token::Integer(0), 4))]);
        assert_lex!(r#"\0624"#, vec![Ok((1, Token::Integer(0o624), 6))]);

        // Octal integer literal followed by non-octal digits.
        assert_lex!(
            r#"\008"#,
            vec![Ok((1, Token::Integer(0), 4)), Ok((4, Token::Integer(8), 5))]
        );
        assert_lex!(
            r#"\01238"#,
            vec![
                Ok((1, Token::Integer(0o123), 6)),
                Ok((6, Token::Integer(8), 7))
            ]
        );
    }

    #[test]
    fn lex_string() {
        assert_lex!(
            r#""this is a string""#,
            vec![Ok((1, Token::String(symbol!("this is a string")), 19,))]
        );

        assert_lex!(
            r#""this is a string"#,
            vec![Err(LexicalError::UnclosedString {
                span: ByteSpan::new(ByteIndex(1), ByteIndex(18))
            })]
        );
    }

    #[test]
    fn lex_whitespace() {
        assert_lex!("      \n \t", vec![]);
        assert_lex!("\r\n", vec![]);
    }
}
