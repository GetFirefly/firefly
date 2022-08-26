use std::collections::VecDeque;
use std::convert::TryFrom;
use std::fmt::Display;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use firefly_diagnostics::{CodeMap, SourceSpan};
use firefly_intern::Symbol;
use firefly_parser::{FileMapSource, Scanner, Source};

use crate::lexer::{AtomToken, SymbolToken, TokenConvertError};
use crate::lexer::{Lexed, Lexer, LexicalToken, Token};

use super::macros::NoArgsMacroCall;
use super::token_stream::TokenStream;
use super::{MacroCall, MacroContainer, PreprocessorError, Result};

pub trait TokenReader: Sized {
    type Source;

    fn new(codemap: Arc<CodeMap>, tokens: Self::Source) -> Self;

    fn inject_include<P>(&mut self, path: P, directive: SourceSpan) -> Result<()>
    where
        P: AsRef<Path>;

    fn read<V: ReadFrom>(&mut self) -> Result<V> {
        V::read_from(self)
    }

    fn try_read<V: ReadFrom>(&mut self) -> Result<Option<V>> {
        V::try_read_from(self)
    }

    fn try_read_macro_call(&mut self, macros: &MacroContainer) -> Result<Option<MacroCall>> {
        if let Some(call) = self.try_read::<NoArgsMacroCall>()? {
            let span = call.span();
            let start = span.start();
            let mut call = MacroCall {
                _question: SymbolToken(start, Token::Question, start),
                name: call.name,
                args: None,
            };

            // The logic for macro resolve/execution is as follows, in order:
            // 1. If there is only a constant macro of the given name and not
            //    a function macro, apply the const without looking ahead.
            // 2. If there exists a function macro with the given name, and
            //    there is a paren open following the macro identifier, then
            //    try to resolve and execute the macro. If there is not a
            //    function macro of the correct arity defined, error.
            // 3. If there is a constant macro, apply that.
            // 4. Error.

            // If there is a function macro defined with the name...
            if macros.defined_func(&call.name()) {
                let paren_ahead = if let Some(lex_tok) = self.try_read_token()? {
                    let res = if let LexicalToken(_, Token::LParen, _) = lex_tok {
                        true
                    } else {
                        false
                    };
                    self.unread_token(lex_tok);
                    res
                } else {
                    false
                };

                // If there is a following LParen, read arguments
                if paren_ahead {
                    call.args = Some(self.read()?);
                }
            }

            Ok(Some(call))
        } else {
            Ok(None)
        }
    }

    fn read_expected<V>(&mut self, expected: &V::Value) -> Result<V>
    where
        V: ReadFrom + Expect + Into<LexicalToken>,
    {
        V::read_expected(self, expected)
    }

    fn try_read_expected<V>(&mut self, expected: &V::Value) -> Result<Option<V>>
    where
        V: ReadFrom + Expect + Into<LexicalToken>,
    {
        V::try_read_expected(self, expected)
    }

    fn try_read_token(&mut self) -> Result<Option<LexicalToken>>;

    fn read_token(&mut self) -> Result<LexicalToken>;

    fn unread_token(&mut self, token: LexicalToken);
}

/// Reads tokens from an in-memory buffer (VecDeque)
pub struct TokenBufferReader {
    codemap: Arc<CodeMap>,
    tokens: VecDeque<Lexed>,
    unread: VecDeque<LexicalToken>,
}
impl TokenReader for TokenBufferReader {
    type Source = VecDeque<Lexed>;

    fn new(codemap: Arc<CodeMap>, tokens: Self::Source) -> Self {
        TokenBufferReader {
            codemap: codemap.clone(),
            tokens,
            unread: VecDeque::new(),
        }
    }

    // Adds tokens from the provided path
    fn inject_include<P>(&mut self, path: P, directive: SourceSpan) -> Result<()>
    where
        P: AsRef<Path>,
    {
        let path = path.as_ref();
        let content =
            std::fs::read_to_string(path).map_err(|source| PreprocessorError::IncludeError {
                source,
                path: path.to_owned(),
                span: directive,
            })?;
        let id = self.codemap.add(path, content);
        let file = self.codemap.get(id).unwrap();
        let source = FileMapSource::new(file);
        let scanner = Scanner::new(source);
        let lexer = Lexer::new(scanner);
        let mut tokens: VecDeque<Lexed> = lexer.collect();
        tokens.append(&mut self.tokens);
        self.tokens = tokens;
        Ok(())
    }

    fn try_read_token(&mut self) -> Result<Option<LexicalToken>> {
        if let Some(token) = self.unread.pop_front() {
            return Ok(Some(token));
        }
        self.tokens
            .pop_front()
            .transpose()
            .map_err(PreprocessorError::from)
    }

    fn read_token(&mut self) -> Result<LexicalToken> {
        if let Some(token) = self.try_read_token()? {
            Ok(token)
        } else {
            Err(PreprocessorError::UnexpectedEOF)
        }
    }

    fn unread_token(&mut self, token: LexicalToken) {
        self.unread.push_front(token);
    }
}

/// Reads tokens from a TokenStream (backed by a Lexer)
pub struct TokenStreamReader<S> {
    codemap: Arc<CodeMap>,
    tokens: TokenStream<S>,
    unread: VecDeque<LexicalToken>,
}
impl<S> TokenReader for TokenStreamReader<S>
where
    S: Source,
{
    type Source = Lexer<S>;

    fn new(codemap: Arc<CodeMap>, tokens: Self::Source) -> Self {
        TokenStreamReader {
            codemap: codemap.clone(),
            tokens: TokenStream::new(tokens),
            unread: VecDeque::new(),
        }
    }

    // Adds tokens from the provided path
    fn inject_include<P>(&mut self, path: P, directive: SourceSpan) -> Result<()>
    where
        P: AsRef<Path>,
    {
        let path = path.as_ref();
        let content =
            fs::read_to_string(path).map_err(|source| PreprocessorError::IncludeError {
                source,
                path: path.to_owned(),
                span: directive,
            })?;
        let id = self.codemap.add_child(path, content, directive);
        let file = self.codemap.get(id).unwrap();
        let source = Source::new(file);
        let scanner = Scanner::new(source);
        let lexer = Lexer::new(scanner);
        self.tokens.include(lexer);
        Ok(())
    }

    fn try_read_token(&mut self) -> Result<Option<LexicalToken>> {
        if let Some(token) = self.unread.pop_front() {
            return Ok(Some(token));
        }
        self.tokens
            .next()
            .transpose()
            .map_err(PreprocessorError::from)
    }

    fn read_token(&mut self) -> Result<LexicalToken> {
        if let Some(token) = self.try_read_token()? {
            Ok(token)
        } else {
            Err(PreprocessorError::UnexpectedEOF)
        }
    }

    fn unread_token(&mut self, token: LexicalToken) {
        self.unread.push_front(token);
    }
}

pub trait ReadFrom: Sized {
    fn read_from<R, S>(reader: &mut R) -> Result<Self>
    where
        R: TokenReader<Source = S>,
    {
        let directive = Self::try_read_from(reader)?;
        Ok(directive.unwrap())
    }

    fn try_read_from<R, S>(reader: &mut R) -> Result<Option<Self>>
    where
        R: TokenReader<Source = S>,
    {
        Self::read_from(reader).map(Some).or_else(|e| match e {
            PreprocessorError::UnexpectedToken { token, .. } => {
                reader.unread_token(token.clone());
                return Ok(None);
            }
            PreprocessorError::InvalidTokenType { token, .. } => {
                reader.unread_token(token.clone());
                return Ok(None);
            }
            PreprocessorError::UnexpectedEOF => {
                return Ok(None);
            }
            _ => Err(e),
        })
    }

    fn read_expected<R, S>(reader: &mut R, expected: &Self::Value) -> Result<Self>
    where
        R: TokenReader<Source = S>,
        Self: Expect + Into<LexicalToken>,
    {
        Self::read_from(reader)
            .map_err(|err| match err {
                PreprocessorError::UnexpectedToken { token, .. } => {
                    PreprocessorError::UnexpectedToken {
                        token,
                        expected: vec![expected.to_string()],
                    }
                }
                PreprocessorError::InvalidTokenType { token, .. } => {
                    PreprocessorError::InvalidTokenType {
                        token,
                        expected: expected.to_string(),
                    }
                }
                _ => err,
            })
            .and_then(|token| {
                if token.expect(expected) {
                    Ok(token)
                } else {
                    Err(PreprocessorError::UnexpectedToken {
                        token: token.into(),
                        expected: vec![expected.to_string()],
                    })
                }
            })
    }

    fn try_read_expected<R, S>(reader: &mut R, expected: &Self::Value) -> Result<Option<Self>>
    where
        R: TokenReader<Source = S>,
        Self: Expect + Into<LexicalToken>,
    {
        Self::try_read_from(reader).map(|token| {
            token.and_then(|token| {
                if token.expect(expected) {
                    Some(token)
                } else {
                    reader.unread_token(token.into());
                    None
                }
            })
        })
    }
}

/// Default implementation for all TryFrom<LexicalToken> supporting types
impl<T> ReadFrom for T
where
    T: TryFrom<LexicalToken, Error = TokenConvertError>,
{
    fn read_from<R, S>(reader: &mut R) -> Result<Self>
    where
        R: TokenReader<Source = S>,
    {
        let token = reader.read_token()?;
        Self::try_from(token).map_err(PreprocessorError::from)
    }
}

pub trait Expect {
    type Value: PartialEq + Display + ?Sized;

    fn expect(&self, expected: &Self::Value) -> bool;
}

impl Expect for AtomToken {
    type Value = Symbol;

    fn expect(&self, expected: &Self::Value) -> bool {
        self.symbol() == *expected
    }
}

impl Expect for SymbolToken {
    type Value = Token;

    fn expect(&self, expected: &Self::Value) -> bool {
        expected.eq(&self.token())
    }
}
