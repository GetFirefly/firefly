//! Module Declarations and Forms
//!
//! See: [6.1 Module Declarations and Forms](http://erlang.org/doc/apps/erts/absform.html#id86691)
use liblumen_beam::serialization::etf;

use super::*;

#[derive(Debug, Clone)]
pub struct ModuleDecl {
    pub forms: Vec<Form>,
}

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
            Self::Module(ref x) => x.line(),
            Self::Behaviour(ref x) => x.line(),
            Self::Export(ref x) => x.line(),
            Self::Import(ref x) => x.line(),
            Self::ExportType(ref x) => x.line(),
            Self::Compile(ref x) => x.line(),
            Self::File(ref x) => x.line(),
            Self::Record(ref x) => x.line(),
            Self::Type(ref x) => x.line(),
            Self::Spec(ref x) => x.line(),
            Self::Attr(ref x) => x.line(),
            Self::Fun(ref x) => x.line(),
            Self::Eof(ref x) => x.line(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Eof {
    pub line: LineNum,
}
impl Eof {
    pub fn new(line: LineNum) -> Self {
        Self { line }
    }
}
impl Node for Eof {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct ModuleAttr {
    pub line: LineNum,
    pub name: String,
}
impl ModuleAttr {
    pub fn new(line: LineNum, name: String) -> Self {
        Self { line, name }
    }
}
impl Node for ModuleAttr {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct BehaviourAttr {
    pub line: LineNum,
    pub is_british: bool,
    pub name: String,
}
impl BehaviourAttr {
    pub fn new(line: LineNum, name: String) -> Self {
        Self {
            line,
            name,
            is_british: true,
        }
    }
    pub fn british(mut self, is_british: bool) -> Self {
        self.is_british = is_british;
        self
    }
}
impl Node for BehaviourAttr {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct ExportAttr {
    pub line: LineNum,
    pub funs: Vec<Export>,
}
impl ExportAttr {
    pub fn new(line: LineNum, funs: Vec<Export>) -> Self {
        Self { line, funs }
    }
}
impl Node for ExportAttr {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct ImportAttr {
    pub line: LineNum,
    pub module: String,
    pub funs: Vec<Import>,
}
impl ImportAttr {
    pub fn new(line: LineNum, module: String, funs: Vec<Import>) -> Self {
        Self { line, module, funs }
    }
}
impl Node for ImportAttr {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct ExportTypeAttr {
    pub line: LineNum,
    pub types: Vec<ExportType>,
}
impl ExportTypeAttr {
    pub fn new(line: LineNum, types: Vec<ExportType>) -> Self {
        Self { line, types }
    }
}
impl Node for ExportTypeAttr {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct CompileOptionsAttr {
    pub line: LineNum,
    pub options: etf::Term,
}
impl CompileOptionsAttr {
    pub fn new(line: LineNum, options: etf::Term) -> Self {
        Self { line, options }
    }
}
impl Node for CompileOptionsAttr {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct FileAttr {
    pub line: LineNum,
    pub original_file: String,
    pub original_line: LineNum,
}
impl FileAttr {
    pub fn new(line: LineNum, original_file: String, original_line: LineNum) -> Self {
        Self {
            line,
            original_file,
            original_line,
        }
    }
}
impl Node for FileAttr {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct RecordDecl {
    pub line: LineNum,
    pub name: String,
    pub fields: Vec<RecordFieldDecl>,
}
impl RecordDecl {
    pub fn new(line: LineNum, name: String, fields: Vec<RecordFieldDecl>) -> Self {
        Self { line, name, fields }
    }
}
impl Node for RecordDecl {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct TypeDecl {
    pub line: LineNum,
    pub is_opaque: bool,
    pub name: String,
    pub vars: Vec<Var>,
    pub ty: Type,
}
impl TypeDecl {
    pub fn new(line: LineNum, name: String, vars: Vec<Var>, ty: Type) -> Self {
        Self {
            line,
            name,
            vars,
            ty,
            is_opaque: false,
        }
    }
    pub fn opaque(mut self, is_opaque: bool) -> Self {
        self.is_opaque = is_opaque;
        self
    }
}
impl Node for TypeDecl {
    fn line(&self) -> LineNum {
        self.line
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
impl FunSpec {
    pub fn new(line: LineNum, name: String, types: Vec<ty::Fun>) -> Self {
        Self {
            line,
            module: None,
            name,
            types,
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
impl Node for FunSpec {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct WildAttr {
    pub line: LineNum,
    pub name: String,
    pub value: etf::Term,
}
impl WildAttr {
    pub fn new(line: LineNum, name: String, value: etf::Term) -> Self {
        Self { line, name, value }
    }
}
impl Node for WildAttr {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct FunDecl {
    pub line: LineNum,
    pub name: String,
    pub clauses: Vec<Clause>,
}
impl FunDecl {
    pub fn new(line: LineNum, name: String, clauses: Vec<Clause>) -> Self {
        Self {
            line,
            name,
            clauses,
        }
    }
}
impl Node for FunDecl {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct RecordFieldDecl {
    pub line: LineNum,
    pub name: String,
    pub ty: Type,
    pub default_value: Expression,
}
impl RecordFieldDecl {
    pub fn new(line: LineNum, name: String) -> Self {
        Self {
            line,
            name,
            ty: Type::any(line),
            default_value: Expression::atom(line, "undefined".to_string()),
        }
    }
    pub fn typ(mut self, ty: ty::Type) -> Self {
        self.ty = ty;
        self
    }
    pub fn default_value(mut self, value: Expression) -> Self {
        self.default_value = value;
        self
    }
}
impl Node for RecordFieldDecl {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct Export {
    pub fun: String,
    pub arity: Arity,
}
impl Export {
    pub fn new(fun: String, arity: Arity) -> Self {
        Self { fun, arity }
    }
}

#[derive(Debug, Clone)]
pub struct Import {
    pub fun: String,
    pub arity: Arity,
}
impl Import {
    pub fn new(fun: String, arity: Arity) -> Self {
        Self { fun, arity }
    }
}

#[derive(Debug, Clone)]
pub struct ExportType {
    pub typ: String,
    pub arity: Arity,
}
impl ExportType {
    pub fn new(typ: String, arity: Arity) -> Self {
        Self { typ, arity }
    }
}
