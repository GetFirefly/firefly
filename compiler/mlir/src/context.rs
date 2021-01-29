use std::convert::AsRef;
use std::ffi::CString;
use std::fmt;
use std::path::Path;
use std::thread::{self, ThreadId};

use anyhow::anyhow;

use liblumen_llvm as llvm;
use liblumen_llvm::enums::{CodeGenOptLevel, CodeGenOptSize};
use liblumen_llvm::target::{TargetMachine, TargetMachineRef};
use liblumen_llvm::utils::{MemoryBuffer, MemoryBufferRef};
use liblumen_session::{Options, OutputType};
use liblumen_util::diagnostics::DiagnosticsHandler;

use crate::{diagnostics, Dialect, Module, ModuleRef};

mod ffi {
    use liblumen_compiler_macros::foreign_struct;

    #[foreign_struct]
    pub struct Context;

    #[foreign_struct]
    pub struct PassManager;
}

pub use self::ffi::{ContextRef, PassManagerRef};

#[repr(C)]
pub struct ContextOptions {
    print_op_on_diagnostic: bool,
    print_stacktrace_on_diagnostic: bool,
    enable_multithreading: bool,
}
impl ContextOptions {
    fn new(options: &Options) -> Self {
        Self {
            print_op_on_diagnostic: options.debugging_opts.print_mlir_op_on_diagnostic,
            print_stacktrace_on_diagnostic: options.debugging_opts.print_mlir_trace_on_diagnostic,
            enable_multithreading: false,
        }
    }
}

pub struct Context {
    thread_id: ThreadId,
    context: ContextRef,
    pass_manager: PassManagerRef,
    target_machine: TargetMachineRef,
    opt: CodeGenOptLevel,
    size: CodeGenOptSize,
    #[allow(dead_code)]
    context_options: ContextOptions,
}
unsafe impl Send for Context {}
unsafe impl Sync for Context {}
impl Context {
    pub fn new(
        thread_id: ThreadId,
        options: &Options,
        diagnostics: &DiagnosticsHandler,
        target_machine: &TargetMachine,
    ) -> Self {
        let target_machine = target_machine.as_ref();
        let (opt, size) = llvm::enums::to_llvm_opt_settings(options.opt_level);
        let context_options = ContextOptions::new(options);
        let context = unsafe {
            let context = MLIRCreateContext(&context_options);
            // Register the MLIR dialects we use
            MLIRRegisterDialects(context);
            // The diagnostics callback expects a reference to our global diagnostics handler
            let handler = diagnostics as *const DiagnosticsHandler;
            diagnostics::MLIRRegisterDiagnosticHandler(
                context,
                handler,
                diagnostics::on_diagnostic,
            );
            context
        };
        let pass_options = PassManagerOptions {
            opt,
            size_opt: size,
            enable_timing: options.debugging_opts.time_passes,
            enable_statistics: options.debugging_opts.perf_stats,
            print_before_pass: options.debugging_opts.print_passes_before,
            print_after_pass: options.debugging_opts.print_passes_after,
            print_module_scope_always: options.debugging_opts.print_mlir_module_scope_always,
            print_after_only_on_change: options.debugging_opts.print_passes_on_change,
        };
        let pass_manager = unsafe { MLIRCreatePassManager(context, target_machine, &pass_options) };
        Self {
            thread_id,
            context,
            pass_manager,
            target_machine,
            opt,
            size,
            context_options,
        }
    }

    pub fn opt_level(&self) -> CodeGenOptLevel {
        self.opt
    }

    pub fn opt_size(&self) -> CodeGenOptSize {
        self.size
    }

    pub fn pass_manager_ref(&self) -> PassManagerRef {
        self.pass_manager
    }

    pub fn target_machine_ref(&self) -> TargetMachineRef {
        self.target_machine
    }

    pub fn parse_file<P: AsRef<Path>>(&self, filename: P) -> anyhow::Result<Module> {
        debug_assert_eq!(
            self.thread_id,
            thread::current().id(),
            "contexts cannot be shared across threads"
        );
        let path = filename.as_ref();
        let file = path.file_name().unwrap().to_string_lossy().into_owned();
        let dialect = if file.ends_with(OutputType::EIRDialect.extension()) {
            Dialect::EIR
        } else if file.ends_with(OutputType::LLVMDialect.extension()) {
            Dialect::LLVM
        } else {
            Dialect::Standard
        };
        let s = path.to_string_lossy().into_owned();
        let f = CString::new(s)?;
        let result = unsafe { MLIRParseFile(self.as_ref(), f.as_ptr()) };
        if result.is_null() {
            Err(anyhow!("failed to parse {}", f.to_string_lossy()))
        } else {
            Ok(Module::new(result, dialect))
        }
    }

    pub fn parse_string<I: AsRef<[u8]>>(&self, name: &str, input: I) -> anyhow::Result<Module> {
        debug_assert_eq!(
            self.thread_id,
            thread::current().id(),
            "contexts cannot be shared across threads"
        );
        let buffer = MemoryBuffer::create_from_slice(input.as_ref(), name);
        let result = unsafe { MLIRParseBuffer(self.as_ref(), buffer.into_mut()) };
        if result.is_null() {
            Err(anyhow!("failed to parse MLIR input"))
        } else {
            Ok(Module::new(result, Dialect::EIR))
        }
    }

    pub fn as_ref(&self) -> ContextRef {
        debug_assert_eq!(
            self.thread_id,
            thread::current().id(),
            "contexts cannot be shared across threads"
        );
        self.context
    }
}
impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MLIRContext({:p})", self.context)
    }
}
impl Eq for Context {}
impl PartialEq for Context {
    fn eq(&self, other: &Self) -> bool {
        self.context == other.context
    }
}

#[repr(C)]
pub struct PassManagerOptions {
    opt: CodeGenOptLevel,
    size_opt: CodeGenOptSize,
    enable_timing: bool,
    enable_statistics: bool,
    print_before_pass: bool,
    print_after_pass: bool,
    print_module_scope_always: bool,
    print_after_only_on_change: bool,
}
impl Default for PassManagerOptions {
    fn default() -> Self {
        Self {
            opt: CodeGenOptLevel::Default,
            size_opt: CodeGenOptSize::Default,
            enable_timing: false,
            enable_statistics: false,
            print_before_pass: false,
            print_after_pass: false,
            print_module_scope_always: false,
            print_after_only_on_change: true,
        }
    }
}

extern "C" {
    pub fn MLIRCreateContext(options: &ContextOptions) -> ContextRef;

    pub fn MLIRRegisterDialects(context: ContextRef);

    pub fn MLIRCreatePassManager(
        context: ContextRef,
        target_machine: TargetMachineRef,
        options: &PassManagerOptions,
    ) -> PassManagerRef;

    pub fn MLIRParseFile(context: ContextRef, filename: *const libc::c_char) -> ModuleRef;

    pub fn MLIRParseBuffer(context: ContextRef, buffer: MemoryBufferRef) -> ModuleRef;
}
