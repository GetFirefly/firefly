#![allow(unused)] // Temporary, remove when done

#[macro_use]
mod macros;
pub(super) mod block;
pub(crate) mod ffi;
pub(super) mod function;
pub(super) mod ops;
pub(super) mod traits;
pub(super) mod value;

use std::cell::RefCell;
use std::collections::HashSet;
use std::convert::AsRef;
use std::ffi::CString;

use anyhow::anyhow;

use log::debug;

use libeir_intern::Symbol;
use libeir_ir as ir;

use liblumen_session::Options;

use crate::mlir::{Context, Module};
use crate::Result;

pub use self::function::{FunctionBuilder, ScopedFunctionBuilder};

pub struct GeneratedModule {
    pub module: Module,
    pub atoms: HashSet<Symbol>,
    pub symbols: HashSet<ir::FunctionIdent>,
}

/// Constructs an MLIR module from an EIR module, using the provided context and options
pub fn build(module: &ir::Module, context: &Context, options: &Options) -> Result<GeneratedModule> {
    debug!("building mlir module for {}", module.name());

    let mut builder = ModuleBuilder::new(module, context);
    return builder.build(options);
}

/// This builder holds the state necessary to build an MLIR module
/// from an EIR module.
///
/// It maintains a module-local atom table, and a table of
/// function symbols created during the build. These are later
/// combined with the same tables of other modules to form a
/// global set of atoms and symbols.
pub struct ModuleBuilder<'m> {
    builder: ffi::ModuleBuilderRef,
    module: &'m ir::Module,
    atoms: RefCell<HashSet<Symbol>>,
    symbols: RefCell<HashSet<ir::FunctionIdent>>,
}
impl<'m> ModuleBuilder<'m> {
    /// Returns the underlying MLIR module builder
    #[inline]
    pub fn as_ref(&self) -> ffi::ModuleBuilderRef {
        self.builder
    }

    /// Creates a new builder for the given EIR module, using the provided MLIR context
    pub fn new(module: &'m ir::Module, context: &Context) -> Self {
        use ffi::MLIRCreateModuleBuilder;

        let name = module.name();
        let c_name = CString::new(name.to_string()).unwrap();
        let builder = unsafe { MLIRCreateModuleBuilder(context.as_ref(), c_name.as_ptr()) };
        Self {
            builder,
            module,
            atoms: RefCell::new(HashSet::new()),
            symbols: RefCell::new(HashSet::new()),
        }
    }

    /// Builds the module by building each function with a FunctionBuilder,
    /// then returning the constructed MLIR module
    ///
    /// Calling this consumes the builder
    pub fn build(mut self, options: &Options) -> Result<GeneratedModule> {
        use ffi::MLIRFinalizeModuleBuilder;

        debug!("building mlir module for {}", self.module.name());

        for f in self.module.function_iter() {
            let mut fb = FunctionBuilder::new(f, &mut self);
            fb.build(options)?;
        }

        debug!("finished building {}, finalizing..", self.module.name());

        let result = unsafe { MLIRFinalizeModuleBuilder(self.builder) };
        if result.is_null() {
            return Err(anyhow!(
                "unexpected error occurred when lowering EIR module"
            ));
        }

        Ok(GeneratedModule {
            module: Module::new(result),
            atoms: self.atoms.into_inner(),
            symbols: self.symbols.into_inner(),
        })
    }

    /// Returns the set of atoms found in this module
    pub fn atoms(&self) -> core::cell::Ref<HashSet<Symbol>> {
        self.atoms.borrow()
    }

    /// Returns the set of atoms found in this module, mutably
    pub fn atoms_mut(&self) -> core::cell::RefMut<HashSet<Symbol>> {
        self.atoms.borrow_mut()
    }

    /// Returns the set of function symbols found in this module
    pub fn symbols(&self) -> core::cell::Ref<HashSet<ir::FunctionIdent>> {
        self.symbols.borrow()
    }

    /// Returns the set of function symbols found in this module, mutably
    pub fn symbols_mut(&self) -> core::cell::RefMut<HashSet<ir::FunctionIdent>> {
        self.symbols.borrow_mut()
    }
}
