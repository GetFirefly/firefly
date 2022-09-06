//! Module Declarations and Forms
//!
//! See: [6.1 Module Declarations and Forms](http://erlang.org/doc/apps/erts/absform.html#id86691)
use firefly_intern::Symbol;

use super::*;

use crate::serialization::etf;

#[derive(Debug, Clone)]
pub enum Form {
    Module(ModuleAttr),
    Behaviour(BehaviourAttr),
    Callback(CallbackAttr),
    OptionalCallbacks(OptionalCallbacksAttr),
    Spec(SpecAttr),
    Export(ExportAttr),
    Import(ImportAttr),
    ExportType(ExportTypeAttr),
    Compile(CompileOptionsAttr),
    File(FileAttr),
    Record(RecordDef),
    Type(TypeDef),
    OnLoad(OnLoadAttr),
    Nifs(NifsAttr),
    Attr(UserAttr),
    Fun(Function),
    Warning(Warning),
    Eof(Eof),
}
impl_from!(Form::Module(ModuleAttr));
impl_from!(Form::Behaviour(BehaviourAttr));
impl_from!(Form::Callback(CallbackAttr));
impl_from!(Form::OptionalCallbacks(OptionalCallbacksAttr));
impl_from!(Form::Spec(SpecAttr));
impl_from!(Form::Export(ExportAttr));
impl_from!(Form::Import(ImportAttr));
impl_from!(Form::ExportType(ExportTypeAttr));
impl_from!(Form::Compile(CompileOptionsAttr));
impl_from!(Form::File(FileAttr));
impl_from!(Form::Record(RecordDef));
impl_from!(Form::Type(TypeDef));
impl_from!(Form::OnLoad(OnLoadAttr));
impl_from!(Form::Nifs(NifsAttr));
impl_from!(Form::Attr(UserAttr));
impl_from!(Form::Fun(Function));
impl_from!(Form::Warning(Warning));
impl_from!(Form::Eof(Eof));
impl Node for Form {
    fn loc(&self) -> Location {
        match self {
            Self::Module(ref x) => x.loc(),
            Self::Behaviour(ref x) => x.loc(),
            Self::Callback(ref x) => x.loc(),
            Self::OptionalCallbacks(ref x) => x.loc(),
            Self::Export(ref x) => x.loc(),
            Self::Import(ref x) => x.loc(),
            Self::ExportType(ref x) => x.loc(),
            Self::Compile(ref x) => x.loc(),
            Self::File(ref x) => x.loc(),
            Self::Record(ref x) => x.loc(),
            Self::Type(ref x) => x.loc(),
            Self::Spec(ref x) => x.loc(),
            Self::OnLoad(ref x) => x.loc(),
            Self::Nifs(ref x) => x.loc(),
            Self::Attr(ref x) => x.loc(),
            Self::Fun(ref x) => x.loc(),
            Self::Warning(ref x) => x.loc(),
            Self::Eof(ref x) => x.loc(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Warning {
    pub loc: Location,
    pub message: etf::Term,
}
impl_node!(Warning);

#[derive(Debug, Clone)]
pub struct Eof {
    pub loc: Location,
}
impl_node!(Eof);
impl Eof {
    pub fn new(loc: Location) -> Self {
        Eof { loc }
    }
}

#[derive(Debug, Clone)]
pub struct ModuleAttr {
    pub loc: Location,
    pub name: Symbol,
}
impl_node!(ModuleAttr);
impl ModuleAttr {
    pub fn new(loc: Location, name: Symbol) -> Self {
        ModuleAttr { loc, name }
    }
}

#[derive(Debug, Clone)]
pub struct BehaviourAttr {
    pub loc: Location,
    pub name: Symbol,
}
impl_node!(BehaviourAttr);
impl BehaviourAttr {
    pub fn new(loc: Location, name: Symbol) -> Self {
        Self { loc, name }
    }
}

#[derive(Debug, Clone)]
pub struct CallbackAttr {
    pub loc: Location,
    pub name: FunctionName,
    pub clauses: Vec<Type>,
}
impl_node!(CallbackAttr);

#[derive(Debug, Clone)]
pub struct OptionalCallbacksAttr {
    pub loc: Location,
    pub funs: Vec<FunctionName>,
}
impl_node!(OptionalCallbacksAttr);
impl OptionalCallbacksAttr {
    pub fn new(loc: Location, funs: Vec<FunctionName>) -> Self {
        Self { loc, funs }
    }
}

#[derive(Debug, Clone)]
pub struct ExportAttr {
    pub loc: Location,
    pub funs: Vec<FunctionName>,
}
impl_node!(ExportAttr);
impl ExportAttr {
    pub fn new(loc: Location, funs: Vec<FunctionName>) -> Self {
        Self { loc, funs }
    }
}

#[derive(Debug, Clone)]
pub struct ImportAttr {
    pub loc: Location,
    pub module: Symbol,
    pub funs: Vec<FunctionName>,
}
impl_node!(ImportAttr);
impl ImportAttr {
    pub fn new(loc: Location, module: Symbol, funs: Vec<FunctionName>) -> Self {
        Self { loc, module, funs }
    }
}

#[derive(Debug, Clone)]
pub struct ExportTypeAttr {
    pub loc: Location,
    pub types: Vec<FunctionName>,
}
impl_node!(ExportTypeAttr);
impl ExportTypeAttr {
    pub fn new(loc: Location, types: Vec<FunctionName>) -> Self {
        Self { loc, types }
    }
}

#[derive(Debug, Clone)]
pub struct OnLoadAttr {
    pub loc: Location,
    pub fun: FunctionName,
}
impl_node!(OnLoadAttr);

#[derive(Debug, Clone)]
pub struct NifsAttr {
    pub loc: Location,
    pub funs: Vec<FunctionName>,
}
impl_node!(NifsAttr);

#[derive(Debug, Clone)]
pub struct CompileOptionsAttr {
    pub loc: Location,
    pub options: Vec<etf::Term>,
}
impl_node!(CompileOptionsAttr);
impl CompileOptionsAttr {
    pub fn new(loc: Location, options: Vec<etf::Term>) -> Self {
        Self { loc, options }
    }
}

#[derive(Debug, Clone)]
pub struct FileAttr {
    pub loc: Location,
    pub original_file: Symbol,
    pub original_line: u32,
}
impl_node!(FileAttr);
impl FileAttr {
    pub fn new(loc: Location, original_file: Symbol, original_line: u32) -> Self {
        Self {
            loc,
            original_file,
            original_line,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RecordDef {
    pub loc: Location,
    pub name: Symbol,
    pub fields: Vec<RecordFieldDef>,
}
impl_node!(RecordDef);
impl RecordDef {
    pub fn new(loc: Location, name: Symbol, fields: Vec<RecordFieldDef>) -> Self {
        Self { loc, name, fields }
    }
}

#[derive(Debug, Clone)]
pub struct TypeDef {
    pub loc: Location,
    pub is_opaque: bool,
    pub name: Symbol,
    pub vars: Vec<Var>,
    pub ty: Type,
}
impl_node!(TypeDef);
impl TypeDef {
    pub fn new(loc: Location, name: Symbol, vars: Vec<Var>, ty: Type) -> Self {
        Self {
            loc,
            name,
            vars,
            ty,
            is_opaque: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpecAttr {
    pub loc: Location,
    pub name: FunctionName,
    pub clauses: Vec<Type>,
}
impl_node!(SpecAttr);
impl SpecAttr {
    pub fn new(loc: Location, name: FunctionName, clauses: Vec<Type>) -> Self {
        Self { loc, name, clauses }
    }
}

#[derive(Debug, Clone)]
pub struct UserAttr {
    pub loc: Location,
    pub name: Symbol,
    pub value: etf::Term,
}
impl_node!(UserAttr);
impl UserAttr {
    pub fn new(loc: Location, name: Symbol, value: etf::Term) -> Self {
        Self { loc, name, value }
    }
}

#[derive(Debug, Clone)]
pub struct Function {
    pub loc: Location,
    pub name: FunctionName,
    pub clauses: Vec<Clause>,
}
impl_node!(Function);
impl Function {
    pub fn new(loc: Location, name: FunctionName, clauses: Vec<Clause>) -> Self {
        Self { loc, name, clauses }
    }
}

#[derive(Debug, Clone)]
pub struct RecordFieldDef {
    pub loc: Location,
    pub name: Symbol,
    pub ty: Type,
    pub default_value: Option<Expression>,
}
impl_node!(RecordFieldDef);
impl RecordFieldDef {
    pub fn new(loc: Location, name: Symbol, ty: Type, default_value: Option<Expression>) -> Self {
        Self {
            loc,
            name,
            ty,
            default_value,
        }
    }
}
