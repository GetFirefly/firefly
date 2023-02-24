use std::convert::AsRef;
use std::fmt;
use std::ops::Deref;
use std::path::Path;

use firefly_llvm::codegen::{CodeGenOptLevel, CodeGenOptSize};
use firefly_session::Options;
use firefly_util::diagnostics::DiagnosticsHandler;

use crate::*;

extern "C" {
    type MlirContext;
}

/// Represents a borrowed reference to an MLIRContext in the FFI bridge
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Context(*mut MlirContext);
impl Context {
    /// Returns true if this is a null reference
    #[inline(always)]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Sets this context to allow unregistered dialects
    pub fn allow_unregistered_dialects(&self) {
        unsafe {
            mlir_context_set_allow_unregistered_dialects(*self, true);
        }
    }

    /// Sets this context to disallow unregistered dialects
    pub fn disallow_unregistered_dialects(&self) {
        unsafe {
            mlir_context_set_allow_unregistered_dialects(*self, false);
        }
    }

    /// Returns the number of dialects registered with the given context.
    /// A registered dialect will be loaded if needed by the parser.
    pub fn num_registered_dialects(&self) -> usize {
        unsafe { mlir_context_get_num_registered_dialects(*self) }
    }

    /// Returns the number of dialects loaded by the context.
    pub fn num_loaded_dialects(&self) -> usize {
        unsafe { mlir_context_get_num_loaded_dialects(*self) }
    }

    /// Gets the dialect instance owned by the given context using the dialect
    /// namespace to identify it, loads (i.e., constructs the instance of) the
    /// dialect if necessary. If the dialect is not registered with the context,
    /// returns null. Use the appropriate `load_<name>_dialect` call to load an
    /// unregistered dialect.
    pub fn get_or_load_dialect<S: Into<StringRef>>(&self, name: S) -> Option<Dialect> {
        let dialect = unsafe { mlir_context_get_or_load_dialect(*self, name.into()) };
        if dialect.is_null() {
            None
        } else {
            Some(dialect)
        }
    }

    /// Set whether or not to print the originating operation when a diagnostic is raised
    pub fn print_op_on_diagnostic(&self, enable: bool) {
        unsafe {
            mlir_context_set_print_op_on_diagnostic(*self, enable);
        }
    }

    /// Set whether or not to print the current stack trace when a diagnostic is raised
    pub fn print_stack_trace_on_diagnostic(&self, enable: bool) {
        unsafe {
            mlir_context_set_print_stack_trace_on_diagnostic(*self, enable);
        }
    }

    /// Set threading mode to multi-threaded.
    ///
    /// NOTE: Cannot use the `print-ir-after-all` flag when multi-threaded
    pub fn enable_multithreading(&self) {
        unsafe {
            mlir_context_enable_multithreading(*self, true);
        }
    }

    /// Set threading mode to single-threaded.
    pub fn disable_multithreading(&self) {
        unsafe {
            mlir_context_enable_multithreading(*self, false);
        }
    }

    /// Returns whether the given fully-qualified operation (i.e.
    /// 'dialect.operation') is registered with the context. This will return true
    /// if the dialect is loaded and the operation is registered within the
    /// dialect.
    pub fn is_registered_operation<S: Into<StringRef>>(&self, name: S) -> bool {
        unsafe { mlir_context_is_registered_operation(*self, name.into()) }
    }
}
impl Eq for Context {}
impl PartialEq for Context {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlir_context_equal(*self, *other) }
    }
}
impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MlirContext({:p})", &self.0)
    }
}
impl fmt::Pointer for Context {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", self.0)
    }
}

/// This type represents an owned reference to an MLIRContext, along
/// with other contextual data needed by the compiler when working with
/// MLIR
pub struct OwnedContext {
    context: Context,
    pass_manager_options: PassManagerOptions,
}
impl Deref for OwnedContext {
    type Target = Context;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.context
    }
}
unsafe impl Send for OwnedContext {}
unsafe impl Sync for OwnedContext {}
impl OwnedContext {
    /// Create a new MLIRContext for the given thread, with the provided options,
    /// and diagnostics handler
    pub fn new(options: &Options, diagnostics: &DiagnosticsHandler) -> Self {
        // Create and configure context
        let context = unsafe { mlir_context_create() };
        context.print_op_on_diagnostic(options.debugging_opts.mlir_print_op_on_diagnostic);
        context
            .print_stack_trace_on_diagnostic(options.debugging_opts.mlir_print_trace_on_diagnostic);
        context.disable_multithreading();
        context.disallow_unregistered_dialects();

        // Register the MLIR dialects we use
        //let llvm_dialect = DialectHandle::get(DialectType::LLVM).unwrap();
        //let func_dialect = DialectHandle::get(DialectType::Func).unwrap();
        //let arith_dialect = DialectHandle::get(DialectType::Arithmetic).unwrap();
        //let cf_dialect = DialectHandle::get(DialectType::ControlFlow).unwrap();
        //let scf_dialect = DialectHandle::get(DialectType::SCF).unwrap();
        //let cir_dialect = DialectHandle::get(DialectType::CIR).unwrap();

        // LLVM requires special registration as its LLVM IR translation interface needs registering as well
        unsafe {
            mlir_context_register_llvm_dialect_translation(context);
        }
        // Register the remaining dialects we use
        //arith_dialect.register(context);
        //func_dialect.register(context);
        //cf_dialect.register(context);
        //scf_dialect.register(context);
        //cir_dialect.register(context);
        // Load the CIR dialect, which will trigger loading of its dependent dialects
        //cir_dialect.load(context);
        //llvm_dialect.load(context);

        // The diagnostics callback expects a reference to our global diagnostics handler
        diagnostics::register_diagnostics_handler(context, diagnostics);

        Self {
            context,
            pass_manager_options: PassManagerOptions::new(options),
        }
    }

    /// Returns the configured speed optimization level
    #[inline(always)]
    pub fn opt_level(&self) -> CodeGenOptLevel {
        self.pass_manager_options.opt_level()
    }

    /// Returns the configured size optimization level
    #[inline(always)]
    pub fn opt_size(&self) -> CodeGenOptSize {
        self.pass_manager_options.size_opt_level()
    }

    /// Returns the current crash reproducer path, if enabled
    pub fn crash_reproducer(&self) -> Option<&Path> {
        self.pass_manager_options.crash_reproducer()
    }

    /// Creates a new PassManager using the options stored in this context
    pub fn create_pass_manager(&self) -> PassManager {
        PassManager::new(self.context, &self.pass_manager_options)
    }

    /// Parse an MLIR module from a file using this context
    pub fn parse_file<P: AsRef<Path>>(&self, filename: P) -> anyhow::Result<OwnedModule> {
        OwnedModule::parse_file(self.context, filename.as_ref())
    }

    /// Parse an MLIR module from a string using this context
    pub fn parse_string<I: AsRef<[u8]>>(&self, input: I) -> anyhow::Result<OwnedModule> {
        OwnedModule::parse_string(self.context, input.as_ref())
    }
}
impl fmt::Debug for OwnedContext {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MlirContext({:p})", self.context)
    }
}
impl Eq for OwnedContext {}
impl PartialEq for OwnedContext {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.context == other.context
    }
}
impl Drop for OwnedContext {
    fn drop(&mut self) {
        unsafe {
            mlir_context_destroy(self.context);
        }
    }
}

extern "C" {
    #[link_name = "mlirContextCreate"]
    fn mlir_context_create() -> Context;
    #[link_name = "mlirContextEqual"]
    fn mlir_context_equal(a: Context, b: Context) -> bool;
    #[link_name = "mlirContextDestroy"]
    fn mlir_context_destroy(context: Context);
    #[link_name = "mlirContextSetPrintOpOnDiagnostic"]
    fn mlir_context_set_print_op_on_diagnostic(context: Context, print: bool);
    #[link_name = "mlirContextSetPrintStackTraceOnDiagnostic"]
    fn mlir_context_set_print_stack_trace_on_diagnostic(context: Context, print: bool);
    #[link_name = "mlirContextEnableMultithreading"]
    fn mlir_context_enable_multithreading(context: Context, enable: bool);
    #[link_name = "mlirContextSetAllowUnregisteredDialects"]
    fn mlir_context_set_allow_unregistered_dialects(context: Context, allow: bool);
    #[link_name = "mlirContextGetNumRegisteredDialects"]
    fn mlir_context_get_num_registered_dialects(context: Context) -> usize;
    #[link_name = "mlirContextGetNumLoadedDialects"]
    fn mlir_context_get_num_loaded_dialects(context: Context) -> usize;
    #[link_name = "mlirContextGetOrLoadDialect"]
    fn mlir_context_get_or_load_dialect(context: Context, name: StringRef) -> Dialect;
    #[link_name = "mlirContextIsRegisteredOperation"]
    fn mlir_context_is_registered_operation(context: Context, name: StringRef) -> bool;
    #[link_name = "mlirContextRegisterLLVMDialectTranslation"]
    fn mlir_context_register_llvm_dialect_translation(context: Context);
}
