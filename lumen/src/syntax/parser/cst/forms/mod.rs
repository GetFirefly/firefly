pub mod parts;

use trackable::track;

use crate::syntax::tokenizer::tokens::{
    AtomToken, IntegerToken, StringToken, SymbolToken, VariableToken,
};
use crate::syntax::tokenizer::values::Symbol;
use crate::syntax::tokenizer::{LexicalToken, Position, PositionRange};

use crate::syntax::parser::traits::{Parse, TokenRead};
use crate::syntax::parser::{Parser, Result};

use super::clauses::{FunDeclClause, SpecClause};
use super::commons::parts::{Args, Clauses, ModulePrefix, NameAndArity};
use super::commons::{ProperList, Tuple};
use super::Type;

use self::parts::RecordFieldDecl;

/// `-` `module` `(` `AtomToken` `)` `.`
#[derive(Debug, Clone)]
pub struct ModuleAttr {
    pub _hyphen: SymbolToken,
    pub _module: AtomToken,
    pub _open: SymbolToken,
    pub module_name: AtomToken,
    pub _close: SymbolToken,
    pub _dot: SymbolToken,
}
impl Parse for ModuleAttr {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        let this = ModuleAttr {
            _hyphen: track!(parser.expect(&Symbol::Hyphen))?,
            _module: track!(parser.expect("module"))?,
            _open: track!(parser.expect(&Symbol::OpenParen))?,
            module_name: track!(parser.parse())?,
            _close: track!(parser.expect(&Symbol::CloseParen))?,
            _dot: track!(parser.expect(&Symbol::Dot))?,
        };
        {
            let module = &this.module_name;
            let reader = parser.reader_mut();
            let module_string = StringToken::from_value(module.value(), module.start_position());
            reader.define_macro("MODULE", vec![module.clone().into()]);
            reader.define_macro("MODULE_STRING", vec![module_string.into()]);
        }
        Ok(this)
    }
}
impl PositionRange for ModuleAttr {
    fn start_position(&self) -> Position {
        self._hyphen.start_position()
    }
    fn end_position(&self) -> Position {
        self._dot.end_position()
    }
}

/// `-` `export` `(` `ProperList<NameAndArity>` `)` `.`
#[derive(Debug, Clone)]
pub struct ExportAttr {
    pub _hyphen: SymbolToken,
    pub _export: AtomToken,
    pub _open: SymbolToken,
    pub exports: ProperList<NameAndArity>,
    pub _close: SymbolToken,
    pub _dot: SymbolToken,
}
impl Parse for ExportAttr {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        Ok(ExportAttr {
            _hyphen: track!(parser.expect(&Symbol::Hyphen))?,
            _export: track!(parser.expect("export"))?,
            _open: track!(parser.expect(&Symbol::OpenParen))?,
            exports: track!(parser.parse())?,
            _close: track!(parser.expect(&Symbol::CloseParen))?,
            _dot: track!(parser.expect(&Symbol::Dot))?,
        })
    }
}
impl PositionRange for ExportAttr {
    fn start_position(&self) -> Position {
        self._hyphen.start_position()
    }
    fn end_position(&self) -> Position {
        self._dot.end_position()
    }
}

/// `-` `export_type` `(` `ProperList<NameAndArity>` `)` `.`
#[derive(Debug, Clone)]
pub struct ExportTypeAttr {
    pub _hyphen: SymbolToken,
    pub _export_type: AtomToken,
    pub _open: SymbolToken,
    pub exports: ProperList<NameAndArity>,
    pub _close: SymbolToken,
    pub _dot: SymbolToken,
}
impl Parse for ExportTypeAttr {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        Ok(ExportTypeAttr {
            _hyphen: track!(parser.expect(&Symbol::Hyphen))?,
            _export_type: track!(parser.expect("export_type"))?,
            _open: track!(parser.expect(&Symbol::OpenParen))?,
            exports: track!(parser.parse())?,
            _close: track!(parser.expect(&Symbol::CloseParen))?,
            _dot: track!(parser.expect(&Symbol::Dot))?,
        })
    }
}
impl PositionRange for ExportTypeAttr {
    fn start_position(&self) -> Position {
        self._hyphen.start_position()
    }
    fn end_position(&self) -> Position {
        self._dot.end_position()
    }
}

/// `-` `import` `(` `AtomToken` `,` `ProperList<NameAndArity>` `)` `.`
#[derive(Debug, Clone)]
pub struct ImportAttr {
    pub _hyphen: SymbolToken,
    pub _import: AtomToken,
    pub _open: SymbolToken,
    pub module_name: AtomToken,
    pub _comma: SymbolToken,
    pub imports: ProperList<NameAndArity>,
    pub _close: SymbolToken,
    pub _dot: SymbolToken,
}
impl Parse for ImportAttr {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        Ok(ImportAttr {
            _hyphen: track!(parser.expect(&Symbol::Hyphen))?,
            _import: track!(parser.expect("import"))?,
            _open: track!(parser.expect(&Symbol::OpenParen))?,
            module_name: track!(parser.parse())?,
            _comma: track!(parser.expect(&Symbol::Comma))?,
            imports: track!(parser.parse())?,
            _close: track!(parser.expect(&Symbol::CloseParen))?,
            _dot: track!(parser.expect(&Symbol::Dot))?,
        })
    }
}
impl PositionRange for ImportAttr {
    fn start_position(&self) -> Position {
        self._hyphen.start_position()
    }
    fn end_position(&self) -> Position {
        self._dot.end_position()
    }
}

/// `-` `file` `(` `StringToken` `,` `IntegerToken` `)` `.`
#[derive(Debug, Clone)]
pub struct FileAttr {
    pub _hyphen: SymbolToken,
    pub _file: AtomToken,
    pub _open: SymbolToken,
    pub file_name: StringToken,
    pub _comma: SymbolToken,
    pub line_num: IntegerToken,
    pub _close: SymbolToken,
    pub _dot: SymbolToken,
}
impl Parse for FileAttr {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        Ok(FileAttr {
            _hyphen: track!(parser.expect(&Symbol::Hyphen))?,
            _file: track!(parser.expect("file"))?,
            _open: track!(parser.expect(&Symbol::OpenParen))?,
            file_name: track!(parser.parse())?,
            _comma: track!(parser.expect(&Symbol::Comma))?,
            line_num: track!(parser.parse())?,
            _close: track!(parser.expect(&Symbol::CloseParen))?,
            _dot: track!(parser.expect(&Symbol::Dot))?,
        })
    }
}
impl PositionRange for FileAttr {
    fn start_position(&self) -> Position {
        self._hyphen.start_position()
    }
    fn end_position(&self) -> Position {
        self._dot.end_position()
    }
}

/// `-` `AtomToken` `(` `Vec<LexicalToken>` `)` `.`
#[derive(Debug, Clone)]
pub struct WildAttr {
    pub _hyphen: SymbolToken,
    pub attr_name: AtomToken,
    pub _open: SymbolToken,
    pub attr_value: Vec<LexicalToken>,
    pub _close: SymbolToken,
    pub _dot: SymbolToken,
}
impl Parse for WildAttr {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        let _hyphen = track!(parser.expect(&Symbol::Hyphen))?;
        let attr_name = track!(parser.parse())?;
        let _open = track!(parser.expect(&Symbol::OpenParen))?;

        let count = parser.peek(|parser| {
            for i in 0.. {
                let v = track!(parser.parse::<LexicalToken>())?
                    .as_symbol_token()
                    .map(|t| t.value());
                if v == Some(Symbol::Dot) {
                    use std::cmp;
                    return Ok(cmp::max(i, 1) - 1);
                }
            }
            unreachable!()
        });
        let attr_value = (0..track!(count)?)
            .map(|_| parser.parse().expect("Never fails"))
            .collect();
        Ok(WildAttr {
            _hyphen,
            attr_name,
            _open,
            attr_value,
            _close: track!(parser.expect(&Symbol::CloseParen))?,
            _dot: track!(parser.expect(&Symbol::Dot))?,
        })
    }
}
impl PositionRange for WildAttr {
    fn start_position(&self) -> Position {
        self._hyphen.start_position()
    }
    fn end_position(&self) -> Position {
        self._dot.end_position()
    }
}

/// `-` `spec` `Option<ModulePrefix>` `AtomToken` `Clauses<SpecClause>` `.`
#[derive(Debug, Clone)]
pub struct FunSpec {
    pub _hyphen: SymbolToken,
    pub _spec: AtomToken,
    pub module: Option<ModulePrefix<AtomToken>>,
    pub fun_name: AtomToken,
    pub clauses: Clauses<SpecClause>,
    pub _dot: SymbolToken,
}
impl Parse for FunSpec {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        Ok(FunSpec {
            _hyphen: track!(parser.expect(&Symbol::Hyphen))?,
            _spec: track!(parser.expect("spec"))?,
            module: track!(parser.parse())?,
            fun_name: track!(parser.parse())?,
            clauses: track!(parser.parse())?,
            _dot: track!(parser.expect(&Symbol::Dot))?,
        })
    }
}
impl PositionRange for FunSpec {
    fn start_position(&self) -> Position {
        self._hyphen.start_position()
    }
    fn end_position(&self) -> Position {
        self._dot.end_position()
    }
}

/// `-` `callback` `AtomToken` `Clauses<SpecClause>` `.`
#[derive(Debug, Clone)]
pub struct CallbackSpec {
    pub _hyphen: SymbolToken,
    pub _spec: AtomToken,
    pub callback_name: AtomToken,
    pub clauses: Clauses<SpecClause>,
    pub _dot: SymbolToken,
}
impl Parse for CallbackSpec {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        Ok(CallbackSpec {
            _hyphen: track!(parser.expect(&Symbol::Hyphen))?,
            _spec: track!(parser.expect("callback"))?,
            callback_name: track!(parser.parse())?,
            clauses: track!(parser.parse())?,
            _dot: track!(parser.expect(&Symbol::Dot))?,
        })
    }
}
impl PositionRange for CallbackSpec {
    fn start_position(&self) -> Position {
        self._hyphen.start_position()
    }
    fn end_position(&self) -> Position {
        self._dot.end_position()
    }
}

/// `Clauses<FunDeclClause>` `.`
#[derive(Debug, Clone)]
pub struct FunDecl {
    pub clauses: Clauses<FunDeclClause>,
    pub _dot: SymbolToken,
}
impl Parse for FunDecl {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        Ok(FunDecl {
            clauses: track!(parser.parse())?,
            _dot: track!(parser.expect(&Symbol::Dot))?,
        })
    }
}
impl PositionRange for FunDecl {
    fn start_position(&self) -> Position {
        self.clauses.start_position()
    }
    fn end_position(&self) -> Position {
        self._dot.end_position()
    }
}

/// `-` `record` `(` `AtomToken` `,` `Tuple<RecordFieldDecl>` `)` `.`
#[derive(Debug, Clone)]
pub struct RecordDecl {
    pub _hyphen: SymbolToken,
    pub _record: AtomToken,
    pub _open: SymbolToken,
    pub record_name: AtomToken,
    pub _comma: SymbolToken,
    pub fields: Tuple<RecordFieldDecl>,
    pub _close: SymbolToken,
    pub _dot: SymbolToken,
}
impl Parse for RecordDecl {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        Ok(RecordDecl {
            _hyphen: track!(parser.expect(&Symbol::Hyphen))?,
            _record: track!(parser.expect("record"))?,
            _open: track!(parser.expect(&Symbol::OpenParen))?,
            record_name: track!(parser.parse())?,
            _comma: track!(parser.expect(&Symbol::Comma))?,
            fields: track!(parser.parse())?,
            _close: track!(parser.expect(&Symbol::CloseParen))?,
            _dot: track!(parser.expect(&Symbol::Dot))?,
        })
    }
}
impl PositionRange for RecordDecl {
    fn start_position(&self) -> Position {
        self._hyphen.start_position()
    }
    fn end_position(&self) -> Position {
        self._dot.end_position()
    }
}

/// `-` `type|opaque` `AtomToken` `Args<VariableToken>` `::` `Type` `.`
#[derive(Debug, Clone)]
pub struct TypeDecl {
    pub _hyphen: SymbolToken,
    pub type_kind: AtomToken,
    pub type_name: AtomToken,
    pub variables: Args<VariableToken>,
    pub _double_colon: SymbolToken,
    pub ty: Type,
    pub _dot: SymbolToken,
}
impl Parse for TypeDecl {
    fn parse<T>(parser: &mut Parser<T>) -> Result<Self>
    where
        T: TokenRead,
    {
        Ok(TypeDecl {
            _hyphen: track!(parser.expect(&Symbol::Hyphen))?,
            type_kind: track!(parser.expect_any(&["type", "opaque"]))?,
            type_name: track!(parser.parse())?,
            variables: track!(parser.parse())?,
            _double_colon: track!(parser.expect(&Symbol::DoubleColon))?,
            ty: track!(parser.parse())?,
            _dot: track!(parser.expect(&Symbol::Dot))?,
        })
    }
}
impl PositionRange for TypeDecl {
    fn start_position(&self) -> Position {
        self._hyphen.start_position()
    }
    fn end_position(&self) -> Position {
        self._dot.end_position()
    }
}
