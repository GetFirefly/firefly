use trackable::track;

use crate::syntax::tokenizer::tokens::{AtomToken, SymbolToken};
use crate::syntax::tokenizer::values::Symbol;
use crate::syntax::tokenizer::{LexicalToken, Position, PositionRange};

use super::directives;
use super::token_reader::{ReadFrom, TokenReader};
use super::Result;

/// Macro directive.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "cargo-clippy", allow(large_enum_variant))]
pub enum Directive {
    Include(directives::Include),
    IncludeLib(directives::IncludeLib),
    Define(directives::Define),
    Undef(directives::Undef),
    Ifdef(directives::Ifdef),
    Ifndef(directives::Ifndef),
    Else(directives::Else),
    Endif(directives::Endif),
    Error(directives::Error),
    Warning(directives::Warning),
}
impl PositionRange for Directive {
    fn start_position(&self) -> Position {
        match *self {
            Directive::Include(ref t) => t.start_position(),
            Directive::IncludeLib(ref t) => t.start_position(),
            Directive::Define(ref t) => t.start_position(),
            Directive::Undef(ref t) => t.start_position(),
            Directive::Ifdef(ref t) => t.start_position(),
            Directive::Ifndef(ref t) => t.start_position(),
            Directive::Else(ref t) => t.start_position(),
            Directive::Endif(ref t) => t.start_position(),
            Directive::Error(ref t) => t.start_position(),
            Directive::Warning(ref t) => t.start_position(),
        }
    }
    fn end_position(&self) -> Position {
        match *self {
            Directive::Include(ref t) => t.end_position(),
            Directive::IncludeLib(ref t) => t.end_position(),
            Directive::Define(ref t) => t.end_position(),
            Directive::Undef(ref t) => t.end_position(),
            Directive::Ifdef(ref t) => t.end_position(),
            Directive::Ifndef(ref t) => t.end_position(),
            Directive::Else(ref t) => t.end_position(),
            Directive::Endif(ref t) => t.end_position(),
            Directive::Error(ref t) => t.end_position(),
            Directive::Warning(ref t) => t.end_position(),
        }
    }
}
impl std::fmt::Display for Directive {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            Directive::Include(ref t) => t.fmt(f),
            Directive::IncludeLib(ref t) => t.fmt(f),
            Directive::Define(ref t) => t.fmt(f),
            Directive::Undef(ref t) => t.fmt(f),
            Directive::Ifdef(ref t) => t.fmt(f),
            Directive::Ifndef(ref t) => t.fmt(f),
            Directive::Else(ref t) => t.fmt(f),
            Directive::Endif(ref t) => t.fmt(f),
            Directive::Error(ref t) => t.fmt(f),
            Directive::Warning(ref t) => t.fmt(f),
        }
    }
}
impl ReadFrom for Directive {
    fn try_read_from<T, E>(reader: &mut TokenReader<T, E>) -> Result<Option<Self>>
    where
        T: Iterator<Item = ::std::result::Result<LexicalToken, E>>,
        E: Into<super::Error>,
    {
        let _hyphen: SymbolToken =
            if let Some(_hyphen) = track!(reader.try_read_expected(&Symbol::Hyphen))? {
                _hyphen
            } else {
                return Ok(None);
            };

        let name: AtomToken = if let Some(name) = track!(reader.try_read())? {
            name
        } else {
            reader.unread_token(_hyphen.into());
            return Ok(None);
        };

        reader.unread_token(name.clone().into());
        reader.unread_token(_hyphen.into());
        match name.value() {
            "include" => track!(reader.read()).map(Directive::Include).map(Some),
            "include_lib" => track!(reader.read()).map(Directive::IncludeLib).map(Some),
            "define" => track!(reader.read()).map(Directive::Define).map(Some),
            "undef" => track!(reader.read()).map(Directive::Undef).map(Some),
            "ifdef" => track!(reader.read()).map(Directive::Ifdef).map(Some),
            "ifndef" => track!(reader.read()).map(Directive::Ifndef).map(Some),
            "else" => track!(reader.read()).map(Directive::Else).map(Some),
            "endif" => track!(reader.read()).map(Directive::Endif).map(Some),
            "error" => track!(reader.read()).map(Directive::Error).map(Some),
            "warning" => track!(reader.read()).map(Directive::Warning).map(Some),
            _ => Ok(None),
        }
    }
}
