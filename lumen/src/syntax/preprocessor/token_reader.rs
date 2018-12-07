use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::path::Path;

use trackable::{track, track_assert, track_panic};

use crate::syntax::tokenizer::tokens::{AtomToken, StringToken, SymbolToken, VariableToken};
use crate::syntax::tokenizer::values::Symbol;
use crate::syntax::tokenizer::{Lexer, LexicalToken};

use super::macros::NoArgsMacroCall;
use super::{Error, ErrorKind, MacroCall, MacroDef, Result};

#[derive(Debug)]
pub struct TokenReader<T, E> {
    tokens: T,
    included_tokens: Vec<Lexer<String>>,
    unread: VecDeque<LexicalToken>,
    _phantom: PhantomData<E>,
}
impl<T, E> TokenReader<T, E>
where
    T: Iterator<Item = ::std::result::Result<LexicalToken, E>>,
    E: Into<Error>,
{
    pub fn new(tokens: T) -> Self {
        TokenReader {
            tokens,
            included_tokens: Vec::new(),
            unread: VecDeque::new(),
            _phantom: PhantomData,
        }
    }

    pub fn add_included_text<P: AsRef<Path>>(&mut self, path: P, text: String) {
        let mut lexer = Lexer::new(text);
        lexer.set_filepath(path);
        self.included_tokens.push(lexer);
    }

    pub fn read<V>(&mut self) -> Result<V>
    where
        V: ReadFrom,
    {
        track!(V::read_from(self))
    }
    pub fn try_read<V>(&mut self) -> Result<Option<V>>
    where
        V: ReadFrom,
    {
        track!(V::try_read_from(self))
    }
    pub fn try_read_macro_call(
        &mut self,
        macros: &HashMap<String, MacroDef>,
    ) -> Result<Option<MacroCall>> {
        if let Some(call) = track!(self.try_read::<NoArgsMacroCall>())? {
            let mut call = MacroCall {
                _question: call._question,
                name: call.name,
                args: None,
            };
            if macros
                .get(call.name.value())
                .map_or(false, |m| m.has_variables())
            {
                call.args = Some(track!(self.read())?);
            }
            Ok(Some(call))
        } else {
            Ok(None)
        }
    }
    pub fn read_expected<V>(&mut self, expected: &V::Value) -> Result<V>
    where
        V: ReadFrom + Expect + Into<LexicalToken>,
    {
        track!(V::read_expected(self, expected))
    }
    pub fn try_read_expected<V>(&mut self, expected: &V::Value) -> Result<Option<V>>
    where
        V: ReadFrom + Expect + Into<LexicalToken>,
    {
        track!(V::try_read_expected(self, expected))
    }
    pub fn try_read_token(&mut self) -> Result<Option<LexicalToken>> {
        if let Some(token) = self.unread.pop_front() {
            Ok(Some(token))
        } else if !self.included_tokens.is_empty() {
            match self.included_tokens.last_mut().expect("Never fails").next() {
                None => {
                    self.included_tokens.pop();
                    self.try_read_token()
                }
                Some(Err(e)) => Err(e.into()),
                Some(Ok(t)) => Ok(Some(t)),
            }
        } else {
            match self.tokens.next() {
                None => Ok(None),
                Some(Err(e)) => Err(e.into()),
                Some(Ok(t)) => Ok(Some(t)),
            }
        }
    }
    pub fn read_token(&mut self) -> Result<LexicalToken> {
        if let Some(token) = track!(self.try_read_token())? {
            Ok(token)
        } else {
            track_panic!(ErrorKind::UnexpectedEos);
        }
    }
    pub fn unread_token(&mut self, token: LexicalToken) {
        self.unread.push_front(token);
    }
}

pub trait ReadFrom: Sized {
    fn read_from<T, E>(reader: &mut TokenReader<T, E>) -> Result<Self>
    where
        T: Iterator<Item = ::std::result::Result<LexicalToken, E>>,
        E: Into<Error>,
    {
        if let Some(directive) = track!(Self::try_read_from(reader))? {
            Ok(directive)
        } else {
            track_panic!(ErrorKind::InvalidInput);
        }
    }
    fn try_read_from<T, E>(reader: &mut TokenReader<T, E>) -> Result<Option<Self>>
    where
        T: Iterator<Item = ::std::result::Result<LexicalToken, E>>,
        E: Into<Error>,
    {
        track!(Self::read_from(reader)).map(Some).or_else(|e| {
            if let ErrorKind::UnexpectedToken(ref token) = *e.kind() {
                reader.unread_token(token.clone());
                return Ok(None);
            }
            if let ErrorKind::UnexpectedEos = *e.kind() {
                return Ok(None);
            }
            Err(e)
        })
    }
    fn read_expected<T, E>(reader: &mut TokenReader<T, E>, expected: &Self::Value) -> Result<Self>
    where
        T: Iterator<Item = ::std::result::Result<LexicalToken, E>>,
        E: Into<Error>,
        Self: Expect + Into<LexicalToken>,
    {
        track!(Self::read_from(reader)).and_then(|token| {
            track_assert!(
                token.expect(expected),
                ErrorKind::UnexpectedToken(token.into())
            );
            Ok(token)
        })
    }
    fn try_read_expected<T, E>(
        reader: &mut TokenReader<T, E>,
        expected: &Self::Value,
    ) -> Result<Option<Self>>
    where
        T: Iterator<Item = ::std::result::Result<LexicalToken, E>>,
        E: Into<Error>,
        Self: Expect + Into<LexicalToken>,
    {
        track!(Self::try_read_from(reader)).map(|token| {
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
impl ReadFrom for AtomToken {
    fn read_from<T, E>(reader: &mut TokenReader<T, E>) -> Result<Self>
    where
        T: Iterator<Item = ::std::result::Result<LexicalToken, E>>,
        E: Into<Error>,
    {
        let token = track!(reader.read_token())?;
        token.into_atom_token().map_err(Error::unexpected_token)
    }
}
impl ReadFrom for VariableToken {
    fn read_from<T, E>(reader: &mut TokenReader<T, E>) -> Result<Self>
    where
        T: Iterator<Item = ::std::result::Result<LexicalToken, E>>,
        E: Into<Error>,
    {
        let token = track!(reader.read_token())?;
        token.into_variable_token().map_err(Error::unexpected_token)
    }
}
impl ReadFrom for SymbolToken {
    fn read_from<T, E>(reader: &mut TokenReader<T, E>) -> Result<Self>
    where
        T: Iterator<Item = ::std::result::Result<LexicalToken, E>>,
        E: Into<Error>,
    {
        let token = track!(reader.read_token())?;
        token.into_symbol_token().map_err(Error::unexpected_token)
    }
}
impl ReadFrom for StringToken {
    fn read_from<T, E>(reader: &mut TokenReader<T, E>) -> Result<Self>
    where
        T: Iterator<Item = ::std::result::Result<LexicalToken, E>>,
        E: Into<Error>,
    {
        let token = track!(reader.read_token())?;
        token.into_string_token().map_err(Error::unexpected_token)
    }
}

pub trait Expect {
    type Value: PartialEq + Debug + ?Sized;
    fn expect(&self, expected: &Self::Value) -> bool;
}
impl Expect for AtomToken {
    type Value = str;
    fn expect(&self, expected: &Self::Value) -> bool {
        self.value() == expected
    }
}
impl Expect for SymbolToken {
    type Value = Symbol;
    fn expect(&self, expected: &Self::Value) -> bool {
        self.value() == *expected
    }
}
