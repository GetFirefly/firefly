use trackable::{track, track_panic};

use crate::syntax::tokenizer::tokens::AtomToken;
use crate::syntax::tokenizer::values::Symbol;
use crate::syntax::tokenizer::{LexicalToken, Position, PositionRange};

use crate::syntax::parser::traits::{Parse, TokenRead};
use crate::syntax::parser::{ErrorKind, Parser, Result};

use super::forms;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "cargo-clippy", allow(large_enum_variant))]
pub enum Form {
    ModuleAttr(forms::ModuleAttr),
    ExportAttr(forms::ExportAttr),
    ExportTypeAttr(forms::ExportTypeAttr),
    ImportAttr(forms::ImportAttr),
    FileAttr(forms::FileAttr),
    WildAttr(forms::WildAttr),
    FunSpec(forms::FunSpec),
    CallbackSpec(forms::CallbackSpec),
    FunDecl(forms::FunDecl),
    RecordDecl(forms::RecordDecl),
    TypeDecl(forms::TypeDecl),
}
impl Parse for Form {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        let kind = track!(FormKind::guess(parser))?;
        Ok(match kind {
            FormKind::ModuleAttr => Form::ModuleAttr(track!(parser.parse())?),
            FormKind::ExportAttr => Form::ExportAttr(track!(parser.parse())?),
            FormKind::ExportTypeAttr => Form::ExportTypeAttr(track!(parser.parse())?),
            FormKind::ImportAttr => Form::ImportAttr(track!(parser.parse())?),
            FormKind::FileAttr => Form::FileAttr(track!(parser.parse())?),
            FormKind::WildAttr => Form::WildAttr(track!(parser.parse())?),
            FormKind::FunSpec => Form::FunSpec(track!(parser.parse())?),
            FormKind::CallbackSpec => Form::CallbackSpec(track!(parser.parse())?),
            FormKind::FunDecl => Form::FunDecl(track!(parser.parse())?),
            FormKind::RecordDecl => Form::RecordDecl(track!(parser.parse())?),
            FormKind::TypeDecl => Form::TypeDecl(track!(parser.parse())?),
        })
    }
}
impl PositionRange for Form {
    fn start_position(&self) -> Position {
        match *self {
            Form::ModuleAttr(ref t) => t.start_position(),
            Form::ExportAttr(ref t) => t.start_position(),
            Form::ExportTypeAttr(ref t) => t.start_position(),
            Form::ImportAttr(ref t) => t.start_position(),
            Form::FileAttr(ref t) => t.start_position(),
            Form::WildAttr(ref t) => t.start_position(),
            Form::FunSpec(ref t) => t.start_position(),
            Form::CallbackSpec(ref t) => t.start_position(),
            Form::FunDecl(ref t) => t.start_position(),
            Form::RecordDecl(ref t) => t.start_position(),
            Form::TypeDecl(ref t) => t.start_position(),
        }
    }
    fn end_position(&self) -> Position {
        match *self {
            Form::ModuleAttr(ref t) => t.end_position(),
            Form::ExportAttr(ref t) => t.end_position(),
            Form::ExportTypeAttr(ref t) => t.end_position(),
            Form::ImportAttr(ref t) => t.end_position(),
            Form::FileAttr(ref t) => t.end_position(),
            Form::WildAttr(ref t) => t.end_position(),
            Form::FunSpec(ref t) => t.end_position(),
            Form::CallbackSpec(ref t) => t.end_position(),
            Form::FunDecl(ref t) => t.end_position(),
            Form::RecordDecl(ref t) => t.end_position(),
            Form::TypeDecl(ref t) => t.end_position(),
        }
    }
}

#[derive(Debug)]
pub enum FormKind {
    ModuleAttr,
    ExportAttr,
    ExportTypeAttr,
    ImportAttr,
    FileAttr,
    WildAttr,
    FunSpec,
    CallbackSpec,
    FunDecl,
    RecordDecl,
    TypeDecl,
}
impl FormKind {
    pub fn guess<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        parser.peek(|parser| {
            Ok(match track!(parser.parse())? {
                LexicalToken::Symbol(ref t) if t.value() == Symbol::Hyphen => {
                    match track!(parser.parse::<AtomToken>())?.value() {
                        "module" => FormKind::ModuleAttr,
                        "export" => FormKind::ExportAttr,
                        "export_type" => FormKind::ExportTypeAttr,
                        "import" => FormKind::ImportAttr,
                        "file" => FormKind::FileAttr,
                        "spec" => FormKind::FunSpec,
                        "callback" => FormKind::CallbackSpec,
                        "record" => FormKind::RecordDecl,
                        "type" | "opaque" => FormKind::TypeDecl,
                        _ => FormKind::WildAttr,
                    }
                }
                LexicalToken::Atom(_) => FormKind::FunDecl,
                token => track_panic!(ErrorKind::UnexpectedToken(token)),
            })
        })
    }
}
