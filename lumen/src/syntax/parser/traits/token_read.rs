use trackable::{track, track_panic};

use crate::syntax::parser::{ErrorKind, Result};
use crate::syntax::tokenizer::LexicalToken;

use super::Preprocessor;

pub trait TokenRead: Preprocessor {
    fn try_read_token(&mut self) -> Result<Option<LexicalToken>>;
    fn read_token(&mut self) -> Result<LexicalToken> {
        if let Some(token) = track!(self.try_read_token())? {
            Ok(token)
        } else {
            track_panic!(ErrorKind::UnexpectedEos);
        }
    }
    fn unread_token(&mut self, token: LexicalToken);
}
impl<'a> Preprocessor for &'a mut TokenRead {
    fn define_macro(&mut self, name: &str, replacement: Vec<LexicalToken>) {
        (*self).define_macro(name, replacement);
    }
    fn undef_macro(&mut self, name: &str) {
        (*self).undef_macro(name);
    }
}
impl<'a> TokenRead for &'a mut TokenRead {
    fn try_read_token(&mut self) -> Result<Option<LexicalToken>> {
        (*self).try_read_token()
    }
    fn unread_token(&mut self, token: LexicalToken) {
        (*self).unread_token(token);
    }
}
