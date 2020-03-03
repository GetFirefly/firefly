pub(crate) mod archive;
pub(crate) mod archive_ro;
pub(crate) mod builder;
pub(crate) mod enums;
pub(crate) mod memory_buffer;
pub(crate) mod string;

pub use self::builder::ModuleBuilder;
pub use self::enums::*;

use std::convert::{AsMut, AsRef};
use std::ffi::CString;
use std::fmt;
use std::mem::MaybeUninit;
use std::os;
use std::path::Path;
use std::ptr;

use anyhow::anyhow;

use llvm_sys::target_machine::LLVMCodeGenFileType;

use liblumen_session::{Emit, OutputType};
use liblumen_util as util;

use super::Result;

use self::memory_buffer::MemoryBuffer;
use self::string::LLVMString;

pub type ContextImpl = llvm_sys::LLVMContext;
pub type ContextRef = llvm_sys::prelude::LLVMContextRef;
pub type LLVMTargetMachine = llvm_sys::target_machine::LLVMOpaqueTargetMachine;
pub type TargetMachineRef = llvm_sys::target_machine::LLVMTargetMachineRef;

pub type ModuleImpl = llvm_sys::LLVMModule;
pub type ModuleRef = llvm_sys::prelude::LLVMModuleRef;

pub use llvm_sys::prelude::{LLVMBasicBlockRef, LLVMTypeRef, LLVMValueRef};
pub use llvm_sys::target::LLVMTargetDataRef;

#[repr(transparent)]
pub struct TargetMachine(TargetMachineRef);
impl TargetMachine {
    pub fn new(ptr: TargetMachineRef) -> Self {
        assert!(!ptr.is_null());
        Self(ptr)
    }

    pub fn as_ref(&self) -> TargetMachineRef {
        self.0
    }

    pub fn get_target_data(&self) -> TargetData {
        use llvm_sys::target_machine::LLVMCreateTargetDataLayout;
        let ptr = unsafe { LLVMCreateTargetDataLayout(self.0) };
        TargetData::new(ptr)
    }
}
unsafe impl Send for TargetMachine {}
unsafe impl Sync for TargetMachine {}
impl Eq for TargetMachine {}
impl PartialEq for TargetMachine {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.0, other.0)
    }
}
impl fmt::Debug for TargetMachine {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "TargetMachine({:p})", self.0)
    }
}

#[repr(transparent)]
pub struct TargetData(LLVMTargetDataRef);
impl TargetData {
    pub fn new(ptr: LLVMTargetDataRef) -> Self {
        assert!(!ptr.is_null());
        Self(ptr)
    }

    pub fn get_pointer_byte_size(&self) -> u32 {
        use llvm_sys::target::LLVMPointerSize;

        unsafe { LLVMPointerSize(self.0) }
    }

    pub fn as_ref(&self) -> LLVMTargetDataRef {
        self.0
    }
}

pub struct Context {
    context: ContextRef,
}
impl Context {
    pub fn new() -> Self {
        let context = unsafe { llvm_sys::core::LLVMContextCreate() };
        Self { context }
    }
}
impl Context {
    pub fn parse_string<I: AsRef<[u8]>>(
        &mut self,
        input: I,
        name: &str,
        tm: TargetMachineRef,
    ) -> Result<Module> {
        let buffer = MemoryBuffer::create_from_slice(input.as_ref(), name);
        self.parse_buffer(buffer, tm)
    }

    pub fn parse_file<P: AsRef<Path>>(
        &mut self,
        filename: P,
        tm: TargetMachineRef,
    ) -> Result<Module> {
        let buffer = MemoryBuffer::create_from_file(filename)?;
        self.parse_buffer(buffer, tm)
    }

    pub fn parse_buffer(
        &mut self,
        mut buffer: MemoryBuffer<'_>,
        tm: TargetMachineRef,
    ) -> Result<Module> {
        use llvm_sys::ir_reader::LLVMParseIRInContext;

        let mut module: *mut ModuleImpl = ptr::null_mut();
        let mut err_string = MaybeUninit::uninit();
        let result = unsafe {
            LLVMParseIRInContext(
                self.context,
                buffer.as_mut(),
                &mut module,
                err_string.as_mut_ptr(),
            )
        };

        if result != 0 {
            let err_string = LLVMString::new(unsafe { err_string.assume_init() });
            return Err(anyhow!("{}", err_string));
        }

        assert!(!module.is_null());

        Ok(Module::new(module, tm))
    }

    pub fn as_ref(&self) -> ContextRef {
        self.context
    }
}
unsafe impl Send for Context {}
unsafe impl Sync for Context {}
//impl Drop for Context {
//    fn drop(&mut self) {
//        unsafe { llvm_sys::core::LLVMContextDispose(self.context); }
//    }
//}
impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LLVMContext({:p})", self.context)
    }
}
impl Eq for Context {}
impl PartialEq for Context {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.context, other.context)
    }
}

#[derive(Clone)]
pub struct Module {
    module: ModuleRef,
    target_machine: TargetMachineRef,
}
unsafe impl Send for Module {}
unsafe impl Sync for Module {}
impl Module {
    pub fn new(module: ModuleRef, target_machine: TargetMachineRef) -> Self {
        assert!(!module.is_null());
        assert!(!target_machine.is_null());
        Self {
            module,
            target_machine,
        }
    }

    pub fn create(name: &str, ctx: &Context, target_machine: TargetMachineRef) -> Result<Self> {
        use llvm_sys::core::{LLVMModuleCreateWithNameInContext, LLVMSetTarget};
        use llvm_sys::target::LLVMSetModuleDataLayout;
        use llvm_sys::target_machine::{LLVMCreateTargetDataLayout, LLVMGetTargetMachineTriple};

        let cstr = CString::new(name).unwrap();
        let m = unsafe { LLVMModuleCreateWithNameInContext(cstr.as_ptr(), ctx.as_ref()) };
        if m.is_null() {
            Err(anyhow!("failed to create LLVM module '{}'", name))
        } else {
            let target_triple = unsafe { LLVMGetTargetMachineTriple(target_machine) };
            let data_layout = unsafe { LLVMCreateTargetDataLayout(target_machine) };

            unsafe {
                LLVMSetTarget(m, target_triple);
            }
            unsafe {
                LLVMSetModuleDataLayout(m, data_layout);
            }

            Ok(Self::new(m, target_machine))
        }
    }

    pub fn dump(&self) {
        use llvm_sys::core::LLVMDumpModule;

        unsafe {
            LLVMDumpModule(self.module);
        }
    }

    /// Emit this module as LLVM IR
    pub fn emit_ir(&self, f: &mut std::fs::File) -> anyhow::Result<()> {
        let fd = util::fs::get_file_descriptor(f);
        let mut err_string = MaybeUninit::uninit();
        let failed = unsafe { LLVMEmitToFileDescriptor(self.module, fd, err_string.as_mut_ptr()) };

        if failed {
            let err_string = LLVMString::new(unsafe { err_string.assume_init() });
            return Err(anyhow!("{}", err_string));
        }

        Ok(())
    }

    /// Emit this module as LLVM bitcode (i.e. 'foo.bc')
    pub fn emit_bc(&self, f: &mut std::fs::File) -> anyhow::Result<()> {
        let fd = util::fs::get_file_descriptor(f);
        let mut err_string = MaybeUninit::uninit();
        let failed =
            unsafe { LLVMEmitBitcodeToFileDescriptor(self.module, fd, err_string.as_mut_ptr()) };

        if failed {
            let err_string = LLVMString::new(unsafe { err_string.assume_init() });
            return Err(anyhow!("{}", err_string));
        }

        Ok(())
    }

    /// Emit this module as (textual) assembly
    pub fn emit_asm(&self, f: &mut std::fs::File) -> anyhow::Result<()> {
        self.emit_file(f, LLVMCodeGenFileType::LLVMAssemblyFile)
    }

    /// Emit this module as a binary object file
    pub fn emit_obj(&self, f: &mut std::fs::File) -> anyhow::Result<()> {
        self.emit_file(f, LLVMCodeGenFileType::LLVMObjectFile)
    }

    fn emit_file(
        &self,
        f: &mut std::fs::File,
        file_type: LLVMCodeGenFileType,
    ) -> anyhow::Result<()> {
        use crate::ffi::target::LLVMTargetMachineEmitToFileDescriptor;

        let fd = util::fs::get_file_descriptor(f);
        let mut err_string = MaybeUninit::uninit();
        let failed = unsafe {
            LLVMTargetMachineEmitToFileDescriptor(
                self.target_machine,
                self.module,
                fd,
                file_type,
                err_string.as_mut_ptr(),
            )
        };

        if failed {
            let err_string = LLVMString::new(unsafe { err_string.assume_init() });
            return Err(anyhow!("{}", err_string));
        }

        Ok(())
    }

    pub fn as_ref(&self) -> ModuleRef {
        self.module
    }

    #[inline]
    pub fn target_machine(&self) -> TargetMachine {
        TargetMachine::new(self.target_machine)
    }
}
impl fmt::Debug for Module {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LLVMModule({:p})", self.module)
    }
}
impl Eq for Module {}
impl PartialEq for Module {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.module, other.module)
    }
}

// The default implementation of `emit` is LLVM IR
impl Emit for Module {
    const TYPE: OutputType = OutputType::LLVMAssembly;

    fn emit(&self, f: &mut std::fs::File) -> anyhow::Result<()> {
        self.emit_ir(f)
    }
}

extern "C" {
    #[cfg(not(windows))]
    pub fn LLVMEmitToFileDescriptor(
        M: ModuleRef,
        fd: os::unix::io::RawFd,
        error_message: *mut *mut libc::c_char,
    ) -> bool;

    #[cfg(windows)]
    pub fn LLVMEmitToFileDescriptor(
        M: ModuleRef,
        fd: os::windows::io::RawHandle,
        error_message: *mut *mut libc::c_char,
    ) -> bool;

    #[cfg(not(windows))]
    pub fn LLVMEmitBitcodeToFileDescriptor(
        M: ModuleRef,
        fd: os::unix::io::RawFd,
        error_message: *mut *mut libc::c_char,
    ) -> bool;

    #[cfg(windows)]
    pub fn LLVMEmitBitcodeToFileDescriptor(
        M: ModuleRef,
        fd: os::windows::io::RawHandle,
        error_message: *mut *mut libc::c_char,
    ) -> bool;
}
