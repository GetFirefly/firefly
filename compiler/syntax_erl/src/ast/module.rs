use std::collections::BTreeMap;
use std::collections::{HashMap, HashSet};

use firefly_syntax_base::*;
use firefly_util::diagnostics::*;
use firefly_util::emit::Emit;

use crate::ast::{self, *};

/// Represents expressions valid at the top level of a module body
#[derive(Debug, Clone, PartialEq, Spanned)]
pub enum TopLevel {
    Module(Ident),
    Attribute(Attribute),
    Record(Record),
    Function(Function),
}
impl TopLevel {
    pub fn module_name(&self) -> Option<Ident> {
        match self {
            Self::Module(name) => Some(*name),
            _ => None,
        }
    }

    pub fn is_attribute(&self) -> bool {
        match self {
            Self::Attribute(_) => true,
            _ => false,
        }
    }

    pub fn is_record(&self) -> bool {
        match self {
            Self::Record(_) => true,
            _ => false,
        }
    }

    pub fn is_function(&self) -> bool {
        match self {
            Self::Function(_) => true,
            _ => false,
        }
    }
}

/// Represents a complete module, broken down into its constituent parts
///
/// Creating a module via `Module::new` ensures that each field is correctly
/// populated, that sanity checking of the top-level constructs is performed,
/// and that a module is ready for semantic analysis and lowering to IR
///
/// A key step performed by `Module::new` is decorating `FunctionVar`
/// structs with the current module where appropriate (as this is currently not
/// done during parsing, as the module is constructed last). This means that once
/// constructed, one can use `FunctionVar` equality in sets/maps, which
/// allows us to easily check definitions, usages, and more.
#[derive(Debug, Clone, Spanned)]
pub struct Module {
    #[span]
    pub span: SourceSpan,
    pub name: Ident,
    pub vsn: Option<ast::Literal>,
    pub author: Option<ast::Literal>,
    pub compile: Option<CompileOptions>,
    pub on_load: Option<Span<FunctionName>>,
    pub nifs: HashSet<Span<FunctionName>>,
    pub imports: HashMap<FunctionName, Span<Signature>>,
    pub exports: HashSet<Span<FunctionName>>,
    pub removed: HashMap<FunctionName, (SourceSpan, Ident)>,
    pub types: HashMap<FunctionName, TypeDef>,
    pub exported_types: HashSet<Span<FunctionName>>,
    pub specs: HashMap<FunctionName, TypeSpec>,
    pub behaviours: HashSet<Ident>,
    pub callbacks: HashMap<FunctionName, Callback>,
    pub records: HashMap<Symbol, Record>,
    pub attributes: HashMap<Ident, ast::Literal>,
    pub functions: BTreeMap<FunctionName, Function>,
    // Used for module-level deprecation
    pub deprecation: Option<Deprecation>,
    // Used for function-level deprecation
    pub deprecations: HashSet<Deprecation>,
}
impl Emit for Module {
    fn file_type(&self) -> Option<&'static str> {
        Some("ast")
    }

    fn emit(&self, f: &mut std::fs::File) -> anyhow::Result<()> {
        use std::io::Write;
        write!(f, "{:#?}", self)?;
        Ok(())
    }
}
impl Module {
    pub fn name(&self) -> Symbol {
        self.name.name
    }

    pub fn record(&self, name: Symbol) -> Option<&Record> {
        self.records.get(&name)
    }

    pub fn is_local(&self, name: &FunctionName) -> bool {
        let local_name = name.to_local();
        self.functions.contains_key(&local_name)
    }

    pub fn is_import(&self, name: &FunctionName) -> bool {
        let local_name = name.to_local();
        !self.is_local(&local_name) && self.imports.contains_key(&local_name)
    }

    /// Creates a new, empty module with the given name and span
    pub fn new(name: Ident, span: SourceSpan) -> Self {
        Self {
            span,
            name,
            vsn: None,
            author: None,
            on_load: None,
            nifs: HashSet::new(),
            compile: None,
            imports: HashMap::new(),
            exports: HashSet::new(),
            removed: HashMap::new(),
            types: HashMap::new(),
            exported_types: HashSet::new(),
            specs: HashMap::new(),
            behaviours: HashSet::new(),
            callbacks: HashMap::new(),
            records: HashMap::new(),
            attributes: HashMap::new(),
            functions: BTreeMap::new(),
            deprecation: None,
            deprecations: HashSet::new(),
        }
    }

    /// Called by the parser to create the module once all of the top-level expressions have been
    /// parsed, in other words this is the last function called when parsing a module.
    ///
    /// As a result, this function performs initial semantic analysis of the module.
    ///
    pub fn new_with_forms(
        diagnostics: &DiagnosticsHandler,
        span: SourceSpan,
        name: Ident,
        mut forms: Vec<TopLevel>,
    ) -> Self {
        use crate::passes::sema;

        let mut module = Self {
            span,
            name,
            vsn: None,
            author: None,
            on_load: None,
            nifs: HashSet::new(),
            compile: None,
            imports: HashMap::new(),
            exports: HashSet::new(),
            removed: HashMap::new(),
            types: HashMap::new(),
            exported_types: HashSet::new(),
            specs: HashMap::new(),

            behaviours: HashSet::new(),
            callbacks: HashMap::new(),
            records: HashMap::new(),
            attributes: HashMap::new(),
            functions: BTreeMap::new(),
            deprecation: None,
            deprecations: HashSet::new(),
        };

        for form in forms.drain(0..) {
            match form {
                TopLevel::Attribute(attr) => {
                    sema::analyze_attribute(diagnostics, &mut module, attr)
                }
                TopLevel::Record(record) => sema::analyze_record(diagnostics, &mut module, record),
                TopLevel::Function(function) => {
                    sema::analyze_function(diagnostics, &mut module, function)
                }
                _ => panic!("unexpected top-level form: {:?}", &form),
            }
        }

        module
    }
}
impl Eq for Module {}
impl PartialEq for Module {
    fn eq(&self, other: &Module) -> bool {
        if self.name != other.name {
            return false;
        }
        if self.vsn != other.vsn {
            return false;
        }
        if self.on_load != other.on_load {
            return false;
        }
        if self.nifs != other.nifs {
            return false;
        }
        if self.imports != other.imports {
            return false;
        }
        if self.exports != other.exports {
            return false;
        }
        if self.types != other.types {
            return false;
        }
        if self.exported_types != other.exported_types {
            return false;
        }
        if self.behaviours != other.behaviours {
            return false;
        }
        if self.callbacks != other.callbacks {
            return false;
        }
        if self.records != other.records {
            return false;
        }
        if self.attributes != other.attributes {
            return false;
        }
        if self.functions != other.functions {
            return false;
        }
        true
    }
}
