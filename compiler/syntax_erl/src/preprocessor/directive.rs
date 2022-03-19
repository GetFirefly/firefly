use std::fmt;

use liblumen_diagnostics::SourceSpan;

use crate::lexer::{AtomToken, LexicalToken, SymbolToken, Token};

use super::directives;
use super::token_reader::{ReadFrom, TokenReader};
use super::Result;

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
    File(directives::File),
}
impl Directive {
    pub fn span(&self) -> SourceSpan {
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
            Directive::File(ref t) => t.span(),
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
            Directive::File(ref t) => t.fmt(f),
        }
    }
}
impl ReadFrom for Directive {
    fn try_read_from<R, S>(reader: &mut R) -> Result<Option<Self>>
    where
        R: TokenReader<Source = S>,
    {
        macro_rules! unread_token {
            ($reader:expr, $hyphen:expr, $source:expr, $tok:expr) => {{
                $reader.unread_token(LexicalToken($source.0, $tok, $source.2));
                $reader.unread_token($hyphen);
                return Ok(None);
            }};
        }

        let _hyphen: SymbolToken = if let Some(_hyphen) = reader.try_read_expected(&Token::Minus)? {
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

        let name_sym = name.symbol().as_str().get();
        // Replace atoms with more concrete tokens for special attributes,
        // but otherwise do nothing else with them
        match name_sym {
            "compile" => unread_token!(reader, _hyphen.into(), name, Token::Compile),
            "record" => unread_token!(reader, _hyphen.into(), name, Token::Record),
            "spec" => unread_token!(reader, _hyphen.into(), name, Token::Spec),
            "callback" => unread_token!(reader, _hyphen.into(), name, Token::Callback),
            "optional_callback" => {
                unread_token!(reader, _hyphen.into(), name, Token::OptionalCallback)
            }
            "import" => unread_token!(reader, _hyphen.into(), name, Token::Import),
            "export" => unread_token!(reader, _hyphen.into(), name, Token::Export),
            "export_type" => unread_token!(reader, _hyphen.into(), name, Token::ExportType),
            "removed" => unread_token!(reader, _hyphen.into(), name, Token::Removed),
            "vsn" => unread_token!(reader, _hyphen.into(), name, Token::Vsn),
            "author" => unread_token!(reader, _hyphen.into(), name, Token::Author),
            "on_load" => unread_token!(reader, _hyphen.into(), name, Token::OnLoad),
            "behaviour" => unread_token!(reader, _hyphen.into(), name, Token::Behaviour),
            "deprecated" => unread_token!(reader, _hyphen.into(), name, Token::Deprecated),
            "type" => unread_token!(reader, _hyphen.into(), name, Token::Type),
            "opaque" => unread_token!(reader, _hyphen.into(), name, Token::Opaque),
            _ => {
                reader.unread_token(name.clone().into());
                reader.unread_token(_hyphen.into());
            }
        }

        match name.symbol().as_str().get() {
            // -module(name) is treated as equivalent to -define(?MODULE, name)
            "module" => reader.read().map(Directive::Module).map(Some),
            // Actual preprocessor directives
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
            "file" => reader.read().map(Directive::File).map(Some),
            _ => Ok(None),
        }
    }
}
