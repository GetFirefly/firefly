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

use liblumen_core::symbols::FunctionSymbol;
use liblumen_session::Options;

use crate::llvm;
use crate::mlir::{Context, Module};
use crate::Result;

pub use self::function::{FunctionBuilder, ScopedFunctionBuilder};

pub struct GeneratedModule {
    pub module: Module,
    pub atoms: HashSet<Symbol>,
    pub symbols: HashSet<FunctionSymbol>,
}

/// Constructs an MLIR module from an EIR module, using the provided context and options
pub fn build(
    module: &ir::Module,
    context: &Context,
    options: &Options,
    target_machine: &llvm::TargetMachine,
) -> Result<GeneratedModule> {
    debug!("building mlir module for {}", module.name());

    let mut builder = ModuleBuilder::new(module, context, target_machine.as_ref());
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
    symbols: RefCell<HashSet<FunctionSymbol>>,
    target_machine: llvm::TargetMachineRef,
}
impl<'m> ModuleBuilder<'m> {
    /// Returns the underlying MLIR module builder
    #[inline]
    pub fn as_ref(&self) -> ffi::ModuleBuilderRef {
        self.builder
    }

    /// Creates a new builder for the given EIR module, using the provided MLIR context
    pub fn new(
        module: &'m ir::Module,
        context: &Context,
        target_machine: llvm::TargetMachineRef,
    ) -> Self {
        use ffi::MLIRCreateModuleBuilder;

        let name = module.name();
        let c_name = CString::new(name.to_string()).unwrap();
        let builder =
            unsafe { MLIRCreateModuleBuilder(context.as_ref(), c_name.as_ptr(), target_machine) };

        let mut atoms = HashSet::new();
        atoms.insert(name.name);

        Self {
            builder,
            module,
            atoms: RefCell::new(atoms),
            symbols: RefCell::new(HashSet::new()),
            target_machine,
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
            let ident = f.function().ident();
            // Don't generate module_info/0 and module_info/1 for now
            if ident.name.name.as_str().get() == "module_info" {
                continue;
            }
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
    pub fn symbols(&self) -> core::cell::Ref<HashSet<FunctionSymbol>> {
        self.symbols.borrow()
    }

    /// Returns the set of function symbols found in this module, mutably
    pub fn symbols_mut(&self) -> core::cell::RefMut<HashSet<FunctionSymbol>> {
        self.symbols.borrow_mut()
    }
}
