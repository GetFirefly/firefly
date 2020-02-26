pub mod builder;
pub use self::builder::GeneratedModule;

use std::cell::RefCell;
use std::convert::AsRef;
use std::ffi::CString;
use std::fmt;
use std::mem::MaybeUninit;
use std::ptr;
use std::os;
use std::path::Path;
use std::thread::{self, ThreadId};

use anyhow::anyhow;

use liblumen_session::{Emit, OutputType};
use liblumen_util as util;

use super::Result;
use crate::ffi::{CodeGenOptLevel, CodeGenOptSize};
use crate::llvm::memory_buffer::{MemoryBuffer, MemoryBufferRef};
use crate::llvm::{self, string::LLVMString, TargetMachineRef};

use self::builder::ffi::{ContextRef, ModuleRef};

/// TargetDialect
#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(C)]
pub enum Dialect {
    #[allow(dead_code)]
    Other,
    None,
    EIR,
    Standard,
    LLVM,
}
impl fmt::Display for Dialect {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut name = format!("{:?}", self);
        name.make_ascii_lowercase();
        write!(f, "{}", &name)
    }
}

pub struct Context {
    context: ContextRef,
    thread_id: ThreadId,
}
unsafe impl Send for Context {}
unsafe impl Sync for Context {}
impl Context {
    pub fn new(thread_id: ThreadId) -> Self {
        let context = unsafe { MLIRCreateContext() };
        Self { context, thread_id }
    }

    pub fn parse_file<P: AsRef<Path>>(&self, filename: P) -> Result<Module> {
        debug_assert_eq!(
            self.thread_id,
            thread::current().id(),
            "contexts cannot be shared across threads"
        );
        let s = filename.as_ref().to_string_lossy().into_owned();
        let f = CString::new(s)?;
        let result = unsafe { MLIRParseFile(self.as_ref(), f.as_ptr()) };
        if result.is_null() {
            Err(anyhow!("failed to parse {}", f.to_string_lossy()))
        } else {
            Ok(Module::new(result))
        }
    }

    pub fn parse_string<I: AsRef<[u8]>>(&self, name: &str, input: I) -> Result<Module> {
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
            Ok(Module::new(result))
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

#[repr(transparent)]
pub struct Module(RefCell<ModuleRef>);
unsafe impl Send for Module {}
unsafe impl Sync for Module {}
impl Module {
    pub fn new(ptr: ModuleRef) -> Self {
        assert!(!ptr.is_null());
        Self(RefCell::new(ptr))
    }

    pub fn lower(&self, context: &Context, dialect: Dialect, opt: CodeGenOptLevel, target_machine: &llvm::TargetMachine) -> Result<()> {
        let result = unsafe { MLIRLowerModule(context.as_ref(), self.as_ref(), dialect, opt, target_machine.as_ref()) };
        if !result.is_null() {
            self.0.replace(result);
            return Ok(());
        }
        Err(anyhow!("lowering to {} failed", dialect))
    }

    pub fn lower_to_llvm_ir(
        &self,
        source_name: Option<String>,
        opt: CodeGenOptLevel,
        size: CodeGenOptSize,
        target_machine: &llvm::TargetMachine,
    ) -> Result<llvm::Module> {
        let result =
            if let Some(sn) = source_name {
                let f = CString::new(sn)?;
                unsafe {
                    MLIRLowerToLLVMIR(
                        self.as_ref(),
                        f.as_ptr(),
                        opt,
                        size,
                        target_machine.as_ref()
                    )
                }
            } else {
                unsafe {
                    MLIRLowerToLLVMIR(
                        self.as_ref(),
                        ptr::null(),
                        opt,
                        size,
                        target_machine.as_ref()
                    )
                }
            };
        if result.is_null() {
            Err(anyhow!("lowering to llvm failed"))
        } else {
            Ok(llvm::Module::new(result, target_machine.as_ref()))
        }
    }

    pub fn as_ref(&self) -> ModuleRef {
        unsafe { *self.0.as_ptr() }
    }
}
impl fmt::Debug for Module {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MLIRModule({:p})", self.as_ref())
    }
}
impl Eq for Module {}
impl PartialEq for Module {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
    }
}
impl Clone for Module {
    fn clone(&self) -> Module {
        Self::new(self.as_ref())
    }
}

impl Emit for Module {
    const TYPE: OutputType = OutputType::EIRDialect;

    fn emit(&self, f: &mut std::fs::File) -> anyhow::Result<()> {
        let fd = util::fs::get_file_descriptor(f);
        let mut err_string = MaybeUninit::uninit();
        let failed =
            unsafe { MLIREmitToFileDescriptor(self.as_ref(), fd, err_string.as_mut_ptr()) };

        if failed {
            let err_string = LLVMString::new(unsafe { err_string.assume_init() });
            return Err(anyhow!("{}", err_string));
        }

        Ok(())
    }
}

extern "C" {
    pub fn MLIRCreateContext() -> ContextRef;

    pub fn MLIRParseFile(context: ContextRef, filename: *const libc::c_char) -> ModuleRef;

    pub fn MLIRParseBuffer(context: ContextRef, buffer: MemoryBufferRef) -> ModuleRef;

    pub fn MLIRLowerModule(
        context: ContextRef,
        module: ModuleRef,
        dialect: Dialect,
        opt: CodeGenOptLevel,
        target_machine: TargetMachineRef,
    ) -> ModuleRef;

    pub fn MLIRLowerToLLVMIR(
        module: ModuleRef,
        source_name: *const libc::c_char,
        opt: CodeGenOptLevel,
        size: CodeGenOptSize,
        target_machine: TargetMachineRef,
    ) -> *mut llvm::ModuleImpl;

    #[cfg(not(windows))]
    pub fn MLIREmitToFileDescriptor(
        M: ModuleRef,
        fd: os::unix::io::RawFd,
        error_message: *mut *mut libc::c_char,
    ) -> bool;

    #[cfg(windows)]
    pub fn MLIREmitToFileDescriptor(
        M: ModuleRef,
        fd: os::windows::io::RawHandle,
        error_message: *mut *mut libc::c_char,
    ) -> bool;

    pub fn MLIREmitToMemoryBuffer(M: ModuleRef) -> llvm::memory_buffer::MemoryBufferRef;
}
