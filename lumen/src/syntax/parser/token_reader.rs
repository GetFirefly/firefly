use std::marker::PhantomData;

use crate::syntax::tokenizer::LexicalToken;

use super::traits::{Preprocessor, TokenRead};
use super::{Error, Result};

#[derive(Debug)]
pub struct TokenReader<T, E> {
    inner: T,
    unread: Vec<LexicalToken>,
    _phantom: PhantomData<E>,
}
impl<T, E> TokenReader<T, E>
where
    T: Iterator<Item = std::result::Result<LexicalToken, E>> + Preprocessor,
    Error: From<E>,
{
    pub fn new(inner: T) -> Self {
        TokenReader {
            inner,
            unread: Vec::new(),
            _phantom: PhantomData,
        }
    }
    pub fn inner(&self) -> &T {
        &self.inner
    }
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }
    pub fn into_inner(self) -> T {
        self.inner
    }
}
impl<T, E> Preprocessor for TokenReader<T, E>
where
    T: Preprocessor,
{
    fn define_macro(&mut self, name: &str, replacement: Vec<LexicalToken>) {
        self.inner.define_macro(name, replacement);
    }
    fn undef_macro(&mut self, name: &str) {
        self.inner.undef_macro(name);
    }
}
impl<T, E> TokenRead for TokenReader<T, E>
where
    T: Iterator<Item = std::result::Result<LexicalToken, E>> + Preprocessor,
    Error: From<E>,
{
    fn try_read_token(&mut self) -> Result<Option<LexicalToken>> {
        match self.unread.pop().map(Ok).or_else(|| self.inner.next()) {
            None => Ok(None),
            Some(Err(e)) => Err(e.into()),
            Some(Ok(t)) => Ok(Some(t)),
        }
    }
    fn unread_token(&mut self, token: LexicalToken) {
        self.unread.push(token);
    }
}
