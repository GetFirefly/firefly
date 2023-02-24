#![deny(warnings)]

mod macros;
pub use self::macros::*;

mod annotations;
pub mod bifs;
mod deprecations;
mod functions;
mod literals;
pub mod nifs;
mod ops;
pub mod printing;
pub mod sets;
mod types;
mod var;

pub use self::annotations::*;
pub use self::deprecations::*;
pub use self::functions::*;
pub use self::literals::{Lit, Literal};
pub use self::ops::*;
pub use self::types::*;
pub use self::var::Var;

use std::collections::{BTreeMap, BTreeSet, HashSet};

use firefly_diagnostics::{SourceSpan, Span};
use firefly_intern::{Ident, Symbol};

/// This structure contains metadata representing an OTP application gathered during parsing and semantic analysis.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ApplicationMetadata {
    pub name: Symbol,
    pub modules: BTreeMap<Symbol, ModuleMetadata>,
}
impl ApplicationMetadata {
    /// Returns the deprecation associated with the given module name, if one was declared
    pub fn get_module_deprecation(&self, name: &Symbol) -> Option<Deprecation> {
        self.modules.get(name).and_then(|m| m.deprecation)
    }

    /// Returns the deprecation associated with the given function name, if one was declared
    pub fn get_function_deprecation(&self, name: &FunctionName) -> Option<Deprecation> {
        let module_name = name.module.unwrap();
        let key = Span::new(SourceSpan::default(), name.to_local());
        let deprecation = self
            .modules
            .get(&module_name)
            .and_then(|m| m.deprecations.get(&key).cloned());
        if deprecation.is_some() {
            deprecation
        } else {
            self.get_module_deprecation(&module_name)
        }
    }
}

/// This structure contains metadata about a module gathered during initial parsing and semantic analysis,
/// which comes in handy during later stages of compilation where we can reason about the set of all
/// reachable modules
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModuleMetadata {
    pub name: Ident,
    pub exports: BTreeSet<Span<FunctionName>>,
    pub deprecation: Option<Deprecation>,
    pub deprecations: BTreeMap<FunctionName, Deprecation>,
}
impl ModuleMetadata {
    pub fn new(name: Ident) -> Self {
        Self {
            name,
            exports: BTreeSet::default(),
            deprecation: None,
            deprecations: BTreeMap::default(),
        }
    }
}

/// This structure holds module-specific compiler options and configuration; it is passed through all phases of
/// compilation alongside its associated module, and is a superset of options in CompilerSettings
/// where applicable
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileOptions {
    // Used to override the filename used in errors/warnings
    pub file: Option<String>,
    pub verbose: bool,
    pub report_errors: bool,
    pub report_warnings: bool,
    // Treats all warnings as errors
    pub warnings_as_errors: bool,
    // Disables warnings
    pub no_warn: bool,
    // Exports all functions
    pub export_all: bool,
    // Prevents auto importing any functions
    pub no_auto_import: bool,
    // Prevents auto importing the specified functions
    pub no_auto_imports: HashSet<FunctionName>,
    // Warns if export_all is used
    pub warn_export_all: bool,
    // Warns when exported variables are used
    pub warn_export_vars: bool,
    // Warns when variables are shadowed
    pub warn_shadow_vars: bool,
    // Warns when a function is unused
    pub warn_unused_function: bool,
    // Disables the unused function warning for the specified functions
    pub no_warn_unused_functions: HashSet<Span<FunctionName>>,
    // Warns about unused imports
    pub warn_unused_import: bool,
    // Warns about unused variables
    pub warn_unused_var: bool,
    // Warns about unused records
    pub warn_unused_record: bool,
    pub warn_untyped_record: bool,
    pub warn_unused_type: bool,
    pub warn_removed: bool,
    pub warn_nif_inline: bool,
    pub warn_bif_clash: bool,
    // Warns about missing type specs
    pub warn_missing_spec: bool,
    pub warn_missing_spec_all: bool,
    pub warn_deprecated_functions: bool,
    pub no_warn_deprecated_functions: HashSet<Span<FunctionName>>,
    pub warn_deprecated_type: bool,
    pub warn_obsolete_guard: bool,
    pub inline: bool,
    // Inlines the given functions
    pub inline_functions: HashSet<Span<FunctionName>>,
}
impl Default for CompileOptions {
    fn default() -> Self {
        CompileOptions {
            file: None,
            verbose: true,
            report_errors: true,
            report_warnings: true,
            warnings_as_errors: false,
            no_warn: false,
            export_all: false,
            no_auto_import: false,
            no_auto_imports: HashSet::new(),
            inline: false,
            inline_functions: HashSet::new(),

            // Warning toggles
            warn_export_all: true,
            warn_export_vars: true,
            warn_shadow_vars: true,
            warn_unused_function: true,
            no_warn_unused_functions: HashSet::new(),
            warn_unused_import: true,
            warn_unused_var: true,
            warn_unused_record: true,
            warn_untyped_record: false,
            warn_unused_type: true,
            warn_removed: true,
            warn_nif_inline: true,
            warn_bif_clash: true,
            warn_missing_spec: false,
            warn_missing_spec_all: false,
            warn_deprecated_functions: true,
            no_warn_deprecated_functions: HashSet::new(),
            warn_deprecated_type: true,
            warn_obsolete_guard: true,
        }
    }
}
