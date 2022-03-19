use std::ffi::c_void;
use std::fmt;
use std::ops::Deref;
use std::path::Path;

use super::*;
use crate::support::{self, MlirStringCallback, StringRef};

extern "C" {
    type MlirPassManager;
    type MlirOpPassManager;
}

/// Represents a borrowed reference to a PassManager in the FFI bridge
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct PassManagerBase(*mut MlirPassManager);
impl PassManagerBase {
    /// Enable printing of IR produced by the pass manager
    pub fn enable_ir_printing_with_flags(self, flags: OpPrintingFlags) {
        unsafe {
            mlir_pass_manager_enable_ir_printing_with_flags(self, &flags);
        }
    }

    /// Enable verification after each pass
    pub fn enable_verifier(self) {
        unsafe {
            mlir_pass_manager_enable_verifier(self, true);
        }
    }

    /// Disable verification after each pass
    pub fn disable_verifier(self) {
        unsafe {
            mlir_pass_manager_enable_verifier(self, false);
        }
    }

    /// Enable timing instrumentation in this pass manager
    fn enable_timing(self) {
        unsafe {
            mlir_pass_manager_enable_timing(self);
        }
    }

    /// Enable statistics instrumentation in this pass manager
    fn enable_statistics(self) {
        unsafe { mlir_pass_manager_enable_statistics(self) }
    }

    /// Enable crash reproducer generation on failure, writing to the given output file
    fn enable_crash_reproducer(self, output_file: &Path) {
        unsafe {
            mlir_pass_manager_enable_crash_reproducer_generation(
                self,
                output_file.try_into().unwrap(),
                /*local_reproducer=*/ true,
            );
        }
    }
}

/// Represents an owned PassManager
///
/// Most functionality is implemented via `PassManagerBase`, but this
/// type gives us ownership over the underlying MLIR `PassManager`
#[repr(transparent)]
pub struct PassManager(PassManagerBase);
impl Deref for PassManager {
    type Target = PassManagerBase;

    #[inline]
    fn deref(&self) -> &PassManagerBase {
        &self.0
    }
}
impl PassManager {
    /// Create a new pass manager using the provided MLIR context
    pub fn new(context: Context, options: &PassManagerOptions) -> Self {
        let pm = unsafe { mlir_pass_manager_create(context) };
        pm.enable_ir_printing_with_flags(options.printing_flags());
        if options.enable_timing {
            pm.enable_timing();
        }
        if options.enable_statistics {
            pm.enable_statistics();
        }
        if options.enable_verifier {
            pm.enable_verifier()
        }
        if let Some(path) = options.enable_crash_reproducer.as_ref() {
            pm.enable_crash_reproducer(path)
        }
        pm
    }

    /// Nest a pass manager for the given operation type under this pass manager
    pub fn nest<'p, 'o: 'p, S: Into<StringRef>>(&'p self, operation_name: S) -> OpPassManager<'o> {
        let inner = unsafe { mlir_pass_manager_get_nested_under(self.0, operation_name.into()) };
        OpPassManager {
            inner,
            _marker: core::marker::PhantomData,
        }
    }

    /// Add an owned pass to this pass manager
    pub fn add<P: Pass>(&self, pass: P) {
        let pass = pass.to_owned();

        unsafe {
            mlir_pass_manager_add_owned_pass(self.0, pass.release());
        }
    }

    /// Run this pass manager against the given MLIR module
    pub fn run<O: Operation>(&mut self, op: &O) -> bool {
        let result = unsafe { mlir_pass_manager_run(self.0, op.base()) };
        result.into()
    }
}
impl<'p, 'o: 'p> Into<OpPassManager<'o>> for &'p PassManager {
    fn into(self) -> OpPassManager<'o> {
        let opm = unsafe { mlir_pass_manager_get_as_op_pass_manager(self.0) };
        OpPassManager {
            inner: opm,
            _marker: core::marker::PhantomData,
        }
    }
}
impl Drop for PassManager {
    fn drop(&mut self) {
        unsafe { mlir_pass_manager_destroy(self.0) }
    }
}
impl fmt::Display for PassManager {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let opm: OpPassManager<'_> = self.into();
        write!(f, "{}", &opm)
    }
}

/// Represents a borrowed reference to an OpPassManager in the FFI bridge
#[repr(transparent)]
#[derive(Copy, Clone)]
struct OpPassManagerBase(*mut MlirOpPassManager);

/// `OpPassManager` is a sub-type of `PassManager` which is used to apply passes
/// to a specific Operation type, wheras `PassManager` is used to apply passes
/// to all operations.
#[repr(transparent)]
pub struct OpPassManager<'a> {
    inner: OpPassManagerBase,
    _marker: core::marker::PhantomData<&'a PassManager>,
}
impl OpPassManager<'_> {
    /// Add an owned pass to this pass manager
    pub fn add<O, P>(&self, pass: P)
    where
        O: Operation,
        P: OpPass<O>,
    {
        let pass = pass.to_owned();
        unsafe {
            mlir_op_pass_manager_add_owned_pass(self.inner, pass.release());
        }
    }

    /// Parse a pass pipeline which will then be managed by this pass manager
    pub fn parse_pipeline(&self, pipeline: &str) -> Result<(), ()> {
        match unsafe { mlir_parse_pass_pipeline(self.inner, pipeline.into()) } {
            LogicalResult::Success => Ok(()),
            _ => Err(()),
        }
    }
}
impl<'o> OpPassManager<'o> {
    /// Create a pass manager for the given operation type under this pass manager
    pub fn nest<'a: 'o>(&self, operation_name: &str) -> OpPassManager<'a> {
        let inner =
            unsafe { mlir_op_pass_manager_get_nested_under(self.inner, operation_name.into()) };
        OpPassManager {
            inner,
            _marker: core::marker::PhantomData,
        }
    }
}
impl fmt::Display for OpPassManager<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe {
            mlir_print_pass_pipeline(
                self.inner,
                support::write_to_formatter,
                f as *mut _ as *mut c_void,
            )
        }
        Ok(())
    }
}

extern "C" {
    #[link_name = "mlirPassManagerCreate"]
    fn mlir_pass_manager_create(context: Context) -> PassManager;

    #[link_name = "mlirPassManagerDestroy"]
    fn mlir_pass_manager_destroy(pm: PassManagerBase);

    #[link_name = "mlirPassManagerEnableIRPrintingWithFlags"]
    fn mlir_pass_manager_enable_ir_printing_with_flags(
        pm: PassManagerBase,
        flags: &OpPrintingFlags,
    );

    #[link_name = "mlirPassManagerEnableVerifier"]
    fn mlir_pass_manager_enable_verifier(pm: PassManagerBase, enable: bool);

    #[link_name = "mlirPassManagerEnableStatistics"]
    fn mlir_pass_manager_enable_statistics(pm: PassManagerBase);

    #[link_name = "mlirPassManagerEnableTiming"]
    fn mlir_pass_manager_enable_timing(pm: PassManagerBase);

    #[link_name = "mlirPassManagerEnableCrashReproducerGeneration"]
    fn mlir_pass_manager_enable_crash_reproducer_generation(
        pm: PassManagerBase,
        output_file: StringRef,
        local_reproducer: bool,
    );

    #[link_name = "mlirPassManagerGetAsOpPassManager"]
    fn mlir_pass_manager_get_as_op_pass_manager(pm: PassManagerBase) -> OpPassManagerBase;

    #[link_name = "mlirPassManagerRun"]
    fn mlir_pass_manager_run(pm: PassManagerBase, op: OperationBase) -> LogicalResult;

    #[link_name = "mlirPassManagerGetNestedUnder"]
    fn mlir_pass_manager_get_nested_under(
        pm: PassManagerBase,
        operation_name: StringRef,
    ) -> OpPassManagerBase;

    #[link_name = "mlirOpPassManagerGetNestedUnder"]
    fn mlir_op_pass_manager_get_nested_under(
        pm: OpPassManagerBase,
        operation_name: StringRef,
    ) -> OpPassManagerBase;

    #[link_name = "mlirPassManagerAddOwnedPass"]
    fn mlir_pass_manager_add_owned_pass(pm: PassManagerBase, pass: PassBase);

    #[link_name = "mlirOpPassManagerAddOwnedPass"]
    fn mlir_op_pass_manager_add_owned_pass(pm: OpPassManagerBase, pass: PassBase);

    #[link_name = "mlirPrintPassPipeline"]
    fn mlir_print_pass_pipeline(
        pm: OpPassManagerBase,
        callback: MlirStringCallback,
        userdata: *const c_void,
    );

    #[link_name = "mlirParsePassPipeline"]
    fn mlir_parse_pass_pipeline(pm: OpPassManagerBase, pipeline: StringRef) -> LogicalResult;
}
