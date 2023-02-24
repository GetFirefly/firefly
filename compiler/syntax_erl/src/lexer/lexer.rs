use std::ops::Range;
use std::str::FromStr;

use firefly_diagnostics::{ByteOffset, SourceIndex, SourceSpan};

use firefly_intern::Symbol;
use firefly_number::{Float, FloatError, Int};
use firefly_parser::{Scanner, Source};

use firefly_parser::{EscapeStm, EscapeStmAction};

use super::errors::LexicalError;
use super::token::*;
use super::Lexed;

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
    /// The location produces is a SourceIndex
    scanner: Scanner<S>,

    /// Escape sequence state machine.
    escape: EscapeStm<SourceIndex>,

    /// The most recent token to be lexed.
    /// At the start and end, this should be Token::EOF
    token: Token,

    /// The position in the input where the current token starts
    /// At the start this will be the byte index of the beginning of the input
    token_start: SourceIndex,

    /// The position in the input where the current token ends
    /// At the start this will be the byte index of the beginning of the input
    token_end: SourceIndex,

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
            escape: EscapeStm::new(),
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

    #[inline]
    fn pop(&mut self) -> char {
        let (pos, c) = self.scanner.pop();
        self.token_end = pos + ByteOffset::from_char_len(c);
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
    fn index(&mut self) -> SourceIndex {
        self.scanner.read().0
    }

    #[inline]
    fn skip(&mut self) {
        self.pop();
    }

    /// Get the span for the current token in `Source`.
    #[inline]
    pub fn span(&self) -> SourceSpan {
        SourceSpan::new(self.token_start, self.token_end)
    }

    /// Get a string slice of the current token.
    #[inline]
    fn slice(&self) -> &str {
        self.scanner.slice(self.span())
    }

    #[inline]
    fn slice_span(&self, span: impl Into<Range<usize>>) -> &str {
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

    #[inline]
    fn skip_whitespace_on_line(&mut self) {
        let mut c: char;
        loop {
            c = self.read();

            if c == '\n' {
                break;
            }

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
            '0'..='9' => self.lex_number(),
            'a'..='z' => self.lex_bare_atom(),
            'A'..='Z' => self.lex_identifier(),
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
                _ => pop!(self, Token::Minus),
            },
            '$' => {
                self.skip();
                if self.read() == '\\' {
                    return match self.lex_escape_sequence() {
                        Ok(num) => match std::char::from_u32(num as u32) {
                            Some(c) => Token::Char(c),
                            None => Token::Integer((num as i64).into()),
                        },
                        Err(err) => Token::Error(err),
                    };
                }
                Token::Char(self.pop())
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
                '<' => pop2!(self, Token::IsLessThanOrEqual),
                ':' => {
                    if self.peek_next() == '=' {
                        pop3!(self, Token::IsExactlyEqual)
                    } else {
                        Token::Error(LexicalError::UnexpectedCharacter {
                            start: self.span().start(),
                            found: self.peek_next(),
                        })
                    }
                }
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
                    Ok(t) => Token::Integer((t as i64).into()),
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
        self.skip_whitespace_on_line();

        // See if this is an Edoc tag
        c = self.read();
        if c == '@' {
            if self.peek().is_ascii_alphabetic() {
                self.skip();

                // Get the tag identifier
                self.lex_identifier();
                // Skip any leading whitespace in the value
                self.skip_whitespace_on_line();
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
    fn lex_string(&mut self) -> Token {
        let quote = self.pop();
        debug_assert!(quote == '"' || quote == '\'');
        let mut buf = None;
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

                    let token = Token::String(symbol);
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
                '0'..='9' => self.skip(),
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
                '0'..='9' => self.skip(),
                c if c.is_alphabetic() => self.skip(),
                _ => break,
            }
        }
        Token::from_bare_atom(self.slice())
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

    #[inline]
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

        // If followed by '#', the leading digits were the radix
        //
        // 10#16
        //   ^ interpret leading as radix
        if c == '#' {
            self.skip();
            // Parse in the given radix
            let radix = match num[..].parse::<u32>() {
                Ok(r) => r,
                Err(e) => {
                    return Token::Error(LexicalError::InvalidRadix {
                        span: self.span(),
                        reason: e.to_string(),
                    });
                }
            };
            if radix >= 2 && radix <= 32 {
                let mut num = String::new();

                // If we have a sign, push that to the new integer string
                c = self.read();
                if c.is_digit(radix) {
                    if negative {
                        num.push('-');
                    }
                }

                // Lex the digits themselves
                if let Err(err) = self.lex_digits(radix, false, &mut num) {
                    return Token::Error(err);
                }

                return to_integer_literal(&num, radix);
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

impl<S> Iterator for Lexer<S>
where
    S: Source,
{
    type Item = Lexed;

    fn next(&mut self) -> Option<Self::Item> {
        let mut res = self.lex();
        loop {
            match res {
                Some(Ok(LexicalToken(_, Token::Comment, _))) => {
                    res = self.lex();
                }
                _ => break,
            }
        }
        res
    }
}

// Converts the string literal into either a `i64` or arbitrary precision integer, preferring `i64`.
//
// This function panics if the literal is unparseable due to being invalid for the given radix,
// or containing non-ASCII digits.
fn to_integer_literal(literal: &str, radix: u32) -> Token {
    let int = Int::from_string_radix(literal, radix).unwrap();
    Token::Integer(int)
}

#[cfg(test)]
mod test {
    use firefly_diagnostics::{ByteIndex, CodeMap, SourceIndex, SourceSpan};
    use firefly_intern::Symbol;
    use firefly_number::Float;
    use firefly_parser::{FileMapSource, Scanner, Source};
    use pretty_assertions::assert_eq;

    use crate::lexer::*;

    macro_rules! symbol {
        ($sym:expr) => {
            Symbol::intern($sym)
        };
    }

    macro_rules! assert_lex(
        ($input:expr, $expected:expr) => ({
            let codemap = CodeMap::new();
            let id = codemap.add("nofile", $input.to_string());
            let file = codemap.get(id).unwrap();
            let source = FileMapSource::new(file);
            let scanner = Scanner::new(source);
            let lexer = Lexer::new(scanner);
            let results = lexer.map(|result| {
                match result {
                    Ok(LexicalToken(_start, token, _end)) => {
                        Ok(token)
                    }
                    Err(err) =>  {
                        Err(err)
                    }
                }
            }).collect::<Vec<_>>();
            assert_eq!(results, $expected(id));
        })
    );

    #[test]
    fn lex_symbols() {
        assert_lex!(":", |_| vec![Ok(Token::Colon)]);
        assert_lex!(",", |_| vec![Ok(Token::Comma)]);
        assert_lex!("=", |_| vec![Ok(Token::Equals)]);
    }

    #[test]
    fn lex_comment() {
        assert_lex!("% this is a comment", |_| vec![]);
        assert_lex!("% @author Paul", |_| vec![Ok(Token::Edoc)]);
    }

    macro_rules! f {
        ($float:expr) => {
            Token::Float(Float::new($float).unwrap())
        };
    }

    #[test]
    fn lex_float_literal() {
        // With leading 0
        assert_lex!("0.0", |_| vec![Ok(f!(0.0))]);
        assert_lex!("000051.0", |_| vec![Ok(f!(51.0))]);
        assert_lex!("05162.0", |_| vec![Ok(f!(5162.0))]);
        assert_lex!("099.0", |_| vec![Ok(f!(99.0))]);
        assert_lex!("04624.51235", |_| vec![Ok(f!(4624.51235))]);
        assert_lex!("0.987", |_| vec![Ok(f!(0.987))]);
        assert_lex!("0.55e10", |_| vec![Ok(f!(0.55e10))]);
        assert_lex!("612.0e61", |_| vec![Ok(f!(612e61))]);
        assert_lex!("0.0e-1", |_| vec![Ok(f!(0e-1))]);
        assert_lex!("41.0e+9", |_| vec![Ok(f!(41e+9))]);

        // Without leading 0
        assert_lex!("5162.0", |_| vec![Ok(f!(5162.0))]);
        assert_lex!("99.0", |_| vec![Ok(f!(99.0))]);
        assert_lex!("4624.51235", |_| vec![Ok(f!(4624.51235))]);
        assert_lex!("612.0e61", |_| vec![Ok(f!(612e61))]);
        assert_lex!("41.0e+9", |_| vec![Ok(f!(41e+9))]);

        // With leading negative sign
        assert_lex!("-700.5", |_| vec![Ok(Token::Minus), Ok(f!(700.5))]);
        assert_lex!("-9.0e2", |_| vec![Ok(Token::Minus), Ok(f!(9.0e2))]);
        assert_lex!("-0.5e1", |_| vec![Ok(Token::Minus), Ok(f!(0.5e1))]);
        assert_lex!("-0.0", |_| vec![Ok(Token::Minus), Ok(f!(0.0))]);

        // Underscores
        assert_lex!("12_3.45_6", |_| vec![Ok(f!(123.456))]);
        assert_lex!("1e1_0", |_| vec![Ok(f!(1e10))]);

        assert_lex!("123_.456", |_| vec![
            Ok(Token::Integer(123.into())),
            Ok(Token::Ident(symbol!("_"))),
            Ok(Token::Dot),
            Ok(Token::Integer(456.into())),
        ]);
    }

    #[test]
    fn lex_identifier_or_atom() {
        assert_lex!("_identifier", |_| vec![Ok(Token::Ident(symbol!(
            "_identifier"
        )))]);
        assert_lex!("_Identifier", |_| vec![Ok(Token::Ident(symbol!(
            "_Identifier"
        )))]);
        assert_lex!("identifier", |_| vec![Ok(Token::Atom(symbol!(
            "identifier"
        )))]);
        assert_lex!("Identifier", |_| vec![Ok(Token::Ident(symbol!(
            "Identifier"
        )))]);
        assert_lex!("z0123", |_| vec![Ok(Token::Atom(symbol!("z0123")))]);
        assert_lex!("i_d@e_t0123", |_| vec![Ok(Token::Atom(symbol!(
            "i_d@e_t0123"
        )))]);
    }

    #[test]
    fn lex_integer_literal() {
        // Decimal
        assert_lex!("1", |_| vec![Ok(Token::Integer(1.into()))]);
        assert_lex!("9624", |_| vec![Ok(Token::Integer(9624.into()))]);
        assert_lex!("-1", |_| vec![
            Ok(Token::Minus),
            Ok(Token::Integer(1.into()))
        ]);
        assert_lex!("-9624", |_| vec![
            Ok(Token::Minus),
            Ok(Token::Integer(9624.into()))
        ]);

        // Hexadecimal
        assert_lex!(r#"\x00"#, |_| vec![Ok(Token::Integer(0x0.into()))]);
        assert_lex!(r#"\x{1234FF}"#, |_| vec![Ok(Token::Integer(
            0x1234FF.into()
        ))]);
        assert_lex!("-16#0", |_| vec![
            Ok(Token::Minus),
            Ok(Token::Integer(0x0.into()))
        ]);
        assert_lex!("-16#1234FF", |_| vec![
            Ok(Token::Minus),
            Ok(Token::Integer(0x1234FF.into()))
        ]);

        // Octal
        assert_lex!(r#"\0"#, |_| vec![Ok(Token::Integer(0.into()))]);
        assert_lex!(r#"\624"#, |_| vec![Ok(Token::Integer(0o624.into()))]);
        assert_lex!(r#"\6244"#, |_| vec![
            Ok(Token::Integer(0o624.into())),
            Ok(Token::Integer(4.into()))
        ]);

        // Octal integer literal followed by non-octal digits.
        assert_lex!(r#"\008"#, |_| vec![
            Ok(Token::Integer(0.into())),
            Ok(Token::Integer(8.into()))
        ]);
        assert_lex!(r#"\1238"#, |_| vec![
            Ok(Token::Integer(0o123.into())),
            Ok(Token::Integer(8.into()))
        ]);

        // Underscores
        assert_lex!("123_456", |_| vec![Ok(Token::Integer(123456.into()))]);
        assert_lex!("123_456_789", |_| vec![Ok(Token::Integer(
            123456789.into()
        ))]);
        assert_lex!("1_2", |_| vec![Ok(Token::Integer(12.into()))]);
        assert_lex!("16#123_abc", |_| vec![Ok(Token::Integer(0x123abc.into()))]);
        assert_lex!("10#123_abc", |_| vec![
            Ok(Token::Integer(123.into())),
            Ok(Token::Ident(Symbol::intern("_abc")))
        ]);
        assert_lex!("1_6#abc", |_| vec![Ok(Token::Integer(0xabc.into()))]);
        assert_lex!("1__0", |_| vec![
            Ok(Token::Integer(1.into())),
            Ok(Token::Ident(Symbol::intern("__0")))
        ]);
        assert_lex!("123_", |_| vec![
            Ok(Token::Integer(123.into())),
            Ok(Token::Ident(Symbol::intern("_"))),
        ]);
        assert_lex!("123__", |_| vec![
            Ok(Token::Integer(123.into())),
            Ok(Token::Ident(Symbol::intern("__"))),
        ]);
    }

    #[test]
    fn lex_string() {
        assert_lex!(r#""this is a string""#, |_| vec![Ok(Token::String(
            symbol!("this is a string")
        ))]);

        assert_lex!(r#""this is a string"#, |source_id| vec![Err(
            LexicalError::UnclosedString {
                span: SourceSpan::new(
                    SourceIndex::new(source_id, ByteIndex(0)),
                    SourceIndex::new(source_id, ByteIndex(17))
                )
            }
        )]);
    }

    #[test]
    fn lex_whitespace() {
        assert_lex!("      \n \t", |_| vec![]);
        assert_lex!("\r\n", |_| vec![]);
    }
}
