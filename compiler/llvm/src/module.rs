use std::ffi::CString;
use std::fmt;
use std::mem::MaybeUninit;
use std::os;
use std::ptr;

use anyhow::anyhow;

use llvm_sys::target_machine::LLVMCodeGenFileType;

use liblumen_session::{Emit, OutputType};
use liblumen_util as util;

use crate::context::Context;
use crate::target::{TargetMachine, TargetMachineRef};
use crate::utils::LLVMString;
use crate::Result;

pub type ModuleImpl = llvm_sys::LLVMModule;
pub type ModuleRef = llvm_sys::prelude::LLVMModuleRef;

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
        use crate::target::LLVMTargetMachineEmitToFileDescriptor;

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
