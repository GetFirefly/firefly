//! Module Declarations and Forms
//!
//! See: [6.1 Module Declarations and Forms](http://erlang.org/doc/apps/erts/absform.html#id86691)
use crate::serialization::etf;

use super::clause;
use super::common;
use super::expr;
use super::ty;
use super::{Arity, LineNum, Node};

#[derive(Debug, Clone)]
pub enum Form {
    Module(ModuleAttr),
    Behaviour(BehaviourAttr),
    Export(ExportAttr),
    Import(ImportAttr),
    ExportType(ExportTypeAttr),
    Compile(CompileOptionsAttr),
    File(FileAttr),
    Record(RecordDecl),
    Type(TypeDecl),
    Spec(FunSpec),
    Attr(WildAttr),
    Fun(FunDecl),
    Eof(Eof),
}
impl_from!(Form::Module(ModuleAttr));
impl_from!(Form::Behaviour(BehaviourAttr));
impl_from!(Form::Export(ExportAttr));
impl_from!(Form::Import(ImportAttr));
impl_from!(Form::ExportType(ExportTypeAttr));
impl_from!(Form::Compile(CompileOptionsAttr));
impl_from!(Form::File(FileAttr));
impl_from!(Form::Record(RecordDecl));
impl_from!(Form::Type(TypeDecl));
impl_from!(Form::Spec(FunSpec));
impl_from!(Form::Attr(WildAttr));
impl_from!(Form::Fun(FunDecl));
impl_from!(Form::Eof(Eof));
impl Node for Form {
    fn line(&self) -> LineNum {
        match *self {
            Form::Module(ref x) => x.line(),
            Form::Behaviour(ref x) => x.line(),
            Form::Export(ref x) => x.line(),
            Form::Import(ref x) => x.line(),
            Form::ExportType(ref x) => x.line(),
            Form::Compile(ref x) => x.line(),
            Form::File(ref x) => x.line(),
            Form::Record(ref x) => x.line(),
            Form::Type(ref x) => x.line(),
            Form::Spec(ref x) => x.line(),
            Form::Attr(ref x) => x.line(),
            Form::Fun(ref x) => x.line(),
            Form::Eof(ref x) => x.line(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Eof {
    pub line: LineNum,
}
impl_node!(Eof);
impl Eof {
    pub fn new(line: LineNum) -> Self {
        Eof { line: line }
    }
}

#[derive(Debug, Clone)]
pub struct ModuleAttr {
    pub line: LineNum,
    pub name: String,
}
impl_node!(ModuleAttr);
impl ModuleAttr {
    pub fn new(line: LineNum, name: String) -> Self {
        ModuleAttr {
            line: line,
            name: name,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BehaviourAttr {
    pub line: LineNum,
    pub is_british: bool,
    pub name: String,
}
impl_node!(BehaviourAttr);
impl BehaviourAttr {
    pub fn new(line: LineNum, name: String) -> Self {
        BehaviourAttr {
            line: line,
            name: name,
            is_british: true,
        }
    }
    pub fn british(mut self, is_british: bool) -> Self {
        self.is_british = is_british;
        self
    }
}

#[derive(Debug, Clone)]
pub struct ExportAttr {
    pub line: LineNum,
    pub funs: Vec<Export>,
}
impl_node!(ExportAttr);
impl ExportAttr {
    pub fn new(line: LineNum, funs: Vec<Export>) -> Self {
        ExportAttr {
            line: line,
            funs: funs,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ImportAttr {
    pub line: LineNum,
    pub module: String,
    pub funs: Vec<Import>,
}
impl_node!(ImportAttr);
impl ImportAttr {
    pub fn new(line: LineNum, module: String, funs: Vec<Import>) -> Self {
        ImportAttr {
            line: line,
            module: module,
            funs: funs,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExportTypeAttr {
    pub line: LineNum,
    pub types: Vec<ExportType>,
}
impl_node!(ExportTypeAttr);
impl ExportTypeAttr {
    pub fn new(line: LineNum, types: Vec<ExportType>) -> Self {
        ExportTypeAttr {
            line: line,
            types: types,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompileOptionsAttr {
    pub line: LineNum,
    pub options: etf::Term,
}
impl_node!(CompileOptionsAttr);
impl CompileOptionsAttr {
    pub fn new(line: LineNum, options: etf::Term) -> Self {
        CompileOptionsAttr {
            line: line,
            options: options,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FileAttr {
    pub line: LineNum,
    pub original_file: String,
    pub original_line: LineNum,
}
impl_node!(FileAttr);
impl FileAttr {
    pub fn new(line: LineNum, original_file: String, original_line: LineNum) -> Self {
        FileAttr {
            line: line,
            original_file: original_file,
            original_line: original_line,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RecordDecl {
    pub line: LineNum,
    pub name: String,
    pub fields: Vec<RecordFieldDecl>,
}
impl_node!(RecordDecl);
impl RecordDecl {
    pub fn new(line: LineNum, name: String, fields: Vec<RecordFieldDecl>) -> Self {
        RecordDecl {
            line: line,
            name: name,
            fields: fields,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TypeDecl {
    pub line: LineNum,
    pub is_opaque: bool,
    pub name: String,
    pub vars: Vec<common::Var>,
    pub ty: ty::Type,
}
impl_node!(TypeDecl);
impl TypeDecl {
    pub fn new(line: LineNum, name: String, vars: Vec<common::Var>, ty: ty::Type) -> Self {
        TypeDecl {
            line: line,
            name: name,
            vars: vars,
            ty: ty,
            is_opaque: false,
        }
    }
    pub fn opaque(mut self, is_opaque: bool) -> Self {
        self.is_opaque = is_opaque;
        self
    }
}

#[derive(Debug, Clone)]
pub struct FunSpec {
    pub line: LineNum,
    pub module: Option<String>,
    pub name: String,
    pub types: Vec<ty::Fun>,
    pub is_callback: bool,
}
impl_node!(FunSpec);
impl FunSpec {
    pub fn new(line: LineNum, name: String, types: Vec<ty::Fun>) -> Self {
        FunSpec {
            line: line,
            module: None,
            name: name,
            types: types,
            is_callback: false,
        }
    }
    pub fn module(mut self, module: String) -> Self {
        self.module = Some(module);
        self
    }
    pub fn callback(mut self, is_callback: bool) -> Self {
        self.is_callback = is_callback;
        self
    }
}

#[derive(Debug, Clone)]
pub struct WildAttr {
    pub line: LineNum,
    pub name: String,
    pub value: etf::Term,
}
impl_node!(WildAttr);
impl WildAttr {
    pub fn new(line: LineNum, name: String, value: etf::Term) -> Self {
        WildAttr {
            line: line,
            name: name,
            value: value,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FunDecl {
    pub line: LineNum,
    pub name: String,
    pub clauses: Vec<clause::Clause>,
}
impl_node!(FunDecl);
impl FunDecl {
    pub fn new(line: LineNum, name: String, clauses: Vec<clause::Clause>) -> Self {
        FunDecl {
            line: line,
            name: name,
            clauses: clauses,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RecordFieldDecl {
    pub line: LineNum,
    pub name: String,
    pub ty: ty::Type,
    pub default_value: expr::Expression,
}
impl_node!(RecordFieldDecl);
impl RecordFieldDecl {
    pub fn new(line: LineNum, name: String) -> Self {
        RecordFieldDecl {
            line: line,
            name: name,
            ty: ty::Type::any(line),
            default_value: expr::Expression::atom(line, "undefined".to_string()),
        }
    }
    pub fn typ(mut self, ty: ty::Type) -> Self {
        self.ty = ty;
        self
    }
    pub fn default_value(mut self, value: expr::Expression) -> Self {
        self.default_value = value;
        self
    }
}

#[derive(Debug, Clone)]
pub struct Export {
    pub fun: String,
    pub arity: Arity,
}
impl Export {
    pub fn new(fun: String, arity: Arity) -> Self {
        Export {
            fun: fun,
            arity: arity,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Import {
    pub fun: String,
    pub arity: Arity,
}
impl Import {
    pub fn new(fun: String, arity: Arity) -> Self {
        Import {
            fun: fun,
            arity: arity,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExportType {
    pub typ: String,
    pub arity: Arity,
}
impl ExportType {
    pub fn new(typ: String, arity: Arity) -> Self {
        ExportType {
            typ: typ,
            arity: arity,
        }
    }
}
