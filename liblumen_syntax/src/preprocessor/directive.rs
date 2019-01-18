use std::fmt;

use liblumen_diagnostics::ByteSpan;

use crate::lexer::{Token, AtomToken, SymbolToken};

use super::Result;

use super::directives;
use super::token_reader::{TokenReader, ReadFrom};

/// Macro directive
#[derive(Debug, Clone)]
pub enum Directive {
    Module(directives::Module),
    Include(directives::Include),
    IncludeLib(directives::IncludeLib),
    Define(directives::Define),
    Undef(directives::Undef),
    Ifdef(directives::Ifdef),
    Ifndef(directives::Ifndef),
    If(directives::If),
    Else(directives::Else),
    Elif(directives::Elif),
    Endif(directives::Endif),
    Error(directives::Error),
    Warning(directives::Warning),
}
impl Directive {
    pub fn span(&self) -> ByteSpan {
        match *self {
            Directive::Module(ref t) => t.span(),
            Directive::Include(ref t) => t.span(),
            Directive::IncludeLib(ref t) => t.span(),
            Directive::Define(ref t) => t.span(),
            Directive::Undef(ref t) => t.span(),
            Directive::Ifdef(ref t) => t.span(),
            Directive::Ifndef(ref t) => t.span(),
            Directive::If(ref t) => t.span(),
            Directive::Else(ref t) => t.span(),
            Directive::Elif(ref t) => t.span(),
            Directive::Endif(ref t) => t.span(),
            Directive::Error(ref t) => t.span(),
            Directive::Warning(ref t) => t.span(),
        }
    }
}
impl fmt::Display for Directive {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Directive::Module(ref t) => t.fmt(f),
            Directive::Include(ref t) => t.fmt(f),
            Directive::IncludeLib(ref t) => t.fmt(f),
            Directive::Define(ref t) => t.fmt(f),
            Directive::Undef(ref t) => t.fmt(f),
            Directive::Ifdef(ref t) => t.fmt(f),
            Directive::Ifndef(ref t) => t.fmt(f),
            Directive::If(ref t) => t.fmt(f),
            Directive::Else(ref t) => t.fmt(f),
            Directive::Elif(ref t) => t.fmt(f),
            Directive::Endif(ref t) => t.fmt(f),
            Directive::Error(ref t) => t.fmt(f),
            Directive::Warning(ref t) => t.fmt(f),
        }
    }
}
impl ReadFrom for Directive {
    fn try_read_from<R, S>(reader: &mut R) -> Result<Option<Self>>
    where
        R: TokenReader<Source = S>,
    {
        let _hyphen: SymbolToken =
            if let Some(_hyphen) = reader.try_read_expected(&Token::Minus)? {
                _hyphen
            } else {
                return Ok(None);
            };

        let name: AtomToken = if let Some(name) = reader.try_read()? {
            name
        } else {
            reader.unread_token(_hyphen.into());
            return Ok(None);
        };

        reader.unread_token(name.clone().into());
        reader.unread_token(_hyphen.into());
        match name.symbol().as_str().get() {
            "module" => reader.read().map(Directive::Module).map(Some),
            "include" => reader.read().map(Directive::Include).map(Some),
            "include_lib" => reader.read().map(Directive::IncludeLib).map(Some),
            "define" => reader.read().map(Directive::Define).map(Some),
            "undef" => reader.read().map(Directive::Undef).map(Some),
            "ifdef" => reader.read().map(Directive::Ifdef).map(Some),
            "ifndef" => reader.read().map(Directive::Ifndef).map(Some),
            "if" => reader.read().map(Directive::If).map(Some),
            "else" => reader.read().map(Directive::Else).map(Some),
            "elif" => reader.read().map(Directive::Elif).map(Some),
            "endif" => reader.read().map(Directive::Endif).map(Some),
            "error" => reader.read().map(Directive::Error).map(Some),
            "warning" => reader.read().map(Directive::Warning).map(Some),
            _ => Ok(None)
        }
    }
}
