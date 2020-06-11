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

use crate::{Dialect, Module, ModuleRef};

mod ffi {
    use liblumen_compiler_macros::foreign_struct;

    #[foreign_struct]
    pub struct Context;

    #[foreign_struct]
    pub struct PassManager;
}

pub use self::ffi::{ContextRef, PassManagerRef};

pub struct Context {
    thread_id: ThreadId,
    context: ContextRef,
    pass_manager: PassManagerRef,
    target_machine: TargetMachineRef,
    opt: CodeGenOptLevel,
    size: CodeGenOptSize,
}
unsafe impl Send for Context {}
unsafe impl Sync for Context {}
impl Context {
    pub fn new(
        thread_id: ThreadId,
        options: &Options,
        llvm_context: &llvm::Context,
        target_machine: &TargetMachine,
    ) -> Self {
        let target_machine = target_machine.as_ref();
        let (opt, size) = llvm::enums::to_llvm_opt_settings(options.opt_level);
        let context = unsafe { MLIRCreateContext() };
        unsafe {
            MLIRRegisterDialects(context, llvm_context.as_ref());
        }
        let enable_timing = options.debugging_opts.time_passes;
        let enable_statistics = options.debugging_opts.perf_stats;
        let pass_manager = unsafe {
            MLIRCreatePassManager(
                context,
                target_machine,
                opt,
                enable_timing,
                enable_statistics,
            )
        };
        Self {
            thread_id,
            context,
            pass_manager,
            target_machine,
            opt,
            size,
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

extern "C" {
    pub fn MLIRCreateContext() -> ContextRef;

    pub fn MLIRRegisterDialects(context: ContextRef, llvm_context: llvm::ContextRef);

    pub fn MLIRCreatePassManager(
        context: ContextRef,
        target_machine: TargetMachineRef,
        opt: CodeGenOptLevel,
        enable_timing: bool,
        enable_statistics: bool,
    ) -> PassManagerRef;

    pub fn MLIRParseFile(context: ContextRef, filename: *const libc::c_char) -> ModuleRef;

    pub fn MLIRParseBuffer(context: ContextRef, buffer: MemoryBufferRef) -> ModuleRef;
}
