pub(crate) mod string;
pub(crate) mod memory_buffer;

use std::ptr;
use std::fmt;
use std::mem;
use std::os;
use std::convert::{AsRef, AsMut};
use std::mem::MaybeUninit;

use anyhow::anyhow;
use llvm_sys::target_machine::LLVMCodeGenFileType;

use liblumen_session::{Emit, OutputType};
use liblumen_util as util;

use self::string::LLVMString;

pub type Context = llvm_sys::LLVMContext;
pub type ContextRef = llvm_sys::prelude::LLVMContextRef;
pub type TargetMachine = llvm_sys::target_machine::LLVMOpaqueTargetMachine;
pub type TargetMachineRef = llvm_sys::target_machine::LLVMTargetMachineRef;

pub type ModuleImpl = llvm_sys::LLVMModule;

pub struct Module<'a> {
    module: &'a mut ModuleImpl,
    target_machine: TargetMachineRef,
}
impl<'a> Module<'a> {
    pub fn new(module: &'a mut ModuleImpl, target_machine: TargetMachineRef) -> Self {
        Self {
            module,
            target_machine,
        }
    }

    /// Emit this module as LLVM IR
    pub fn emit_ir(&self, f: &mut std::fs::File) -> anyhow::Result<()> {
        let fd = util::fs::get_file_descriptor(f);
        let mut err_string = MaybeUninit::uninit();
        let failed = unsafe {
            LLVMEmitToFileDescriptor(
                self.module,
                fd,
                err_string.as_mut_ptr()
            )
        };

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

    fn emit_file(&self, f: &mut std::fs::File, file_type: LLVMCodeGenFileType) -> anyhow::Result<()> {
        use crate::ffi::target::LLVMTargetMachineEmitToFileDescriptor;

        let fd = util::fs::get_file_descriptor(f);
        let mut err_string = MaybeUninit::uninit();
        let failed = unsafe {
            LLVMTargetMachineEmitToFileDescriptor(
                self.target_machine,
                self.module,
                fd,
                file_type,
                err_string.as_mut_ptr()
            )
        };

        if failed {
            let err_string = LLVMString::new(unsafe { err_string.assume_init() });
            return Err(anyhow!("{}", err_string));
        }

        Ok(())
    }
}
impl<'a> fmt::Debug for Module<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LLVMModule({:p})", self.module as *const _)
    }
}
impl<'a> Eq for Module<'a> {}
impl<'a> PartialEq for Module<'a> {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.module as *const _, other.module as *const _)
    }
}
impl Clone for Module<'static> {
    fn clone(&self) -> Module<'static> {
        let target_machine = self.target_machine;
        let ptr = self.module as *const _ as *mut ModuleImpl;
        let module = unsafe {
            mem::transmute::<*mut ModuleImpl, &'static mut ModuleImpl>(ptr)
        };
        Self {
            module,
            target_machine,
        }
    }
}
impl<'a> AsRef<ModuleImpl> for Module<'a> {
    fn as_ref(&self) -> &ModuleImpl {
        self.module
    }
}
impl<'a> AsMut<ModuleImpl> for Module<'a> {
    fn as_mut(&mut self) -> &mut ModuleImpl {
        self.module
    }
}

// The default implementation of `emit` is LLVM IR
impl<'a> Emit for Module<'a> {
    const TYPE: OutputType = OutputType::LLVMAssembly;

    fn emit(&self, f: &mut std::fs::File) -> anyhow::Result<()> {
        self.emit_ir(f)
    }
}

extern "C" {
    #[cfg(not(windows))]
    pub fn LLVMEmitToFileDescriptor(
        M: &ModuleImpl,
        fd: os::unix::io::RawFd,
        error_message: *mut *mut libc::c_char
    ) -> bool;

    #[cfg(windows)]
    pub fn LLVMEmitToFileDescriptor(
        M: &ModuleImpl,
        fd: os::windows::io::RawHandle,
        error_message: *mut *mut libc::c_char
    ) -> bool;
}
