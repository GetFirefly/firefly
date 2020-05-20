use std::cell::RefCell;
use std::ffi::CString;
use std::fmt;
use std::mem::MaybeUninit;
use std::os;
use std::ptr;

use anyhow::anyhow;

use liblumen_llvm as llvm;
use liblumen_llvm::enums::{CodeGenOptLevel, CodeGenOptSize};
use liblumen_llvm::target::TargetMachineRef;
use liblumen_llvm::utils::{LLVMString, MemoryBufferRef};
use liblumen_session::{Emit, OutputType};
use liblumen_util as util;

use crate::context::PassManagerRef;

mod ffi {
    use liblumen_compiler_macros::foreign_struct;

    #[foreign_struct]
    pub struct Module;
}

pub use self::ffi::ModuleRef;

use crate::{Context, ContextRef, Dialect};

pub struct Module {
    module: RefCell<ModuleRef>,
    dialect: RefCell<Dialect>,
}
unsafe impl Send for Module {}
unsafe impl Sync for Module {}
impl Module {
    pub fn new(ptr: ModuleRef, dialect: Dialect) -> Self {
        assert!(!ptr.is_null());
        Self {
            module: RefCell::new(ptr),
            dialect: RefCell::new(dialect),
        }
    }

    pub fn lower(&self, context: &Context) -> anyhow::Result<()> {
        let pass_manager = context.pass_manager_ref();
        let target_machine = context.target_machine_ref();
        let result = unsafe {
            MLIRLowerModule(
                context.as_ref(),
                pass_manager,
                target_machine,
                self.as_ref(),
            )
        };
        if !result.is_null() {
            self.module.replace(result);
            self.dialect.replace(Dialect::LLVM);
            return Ok(());
        }
        Err(anyhow!("lowering to mlir (llvm dialect) failed"))
    }

    pub fn lower_to_llvm_ir(
        &self,
        context: &Context,
        source_name: Option<String>,
    ) -> anyhow::Result<llvm::module::Module> {
        let target_machine = context.target_machine_ref();
        let result = if let Some(sn) = source_name {
            let source_name_bytes = sn.as_bytes();
            let source_name = source_name_bytes.as_ptr();
            let source_name_len = source_name_bytes.len();
            unsafe {
                MLIRLowerToLLVMIR(
                    self.as_ref(),
                    target_machine,
                    source_name as *const libc::c_char,
                    source_name_len as libc::c_uint,
                )
            }
        } else {
            unsafe { MLIRLowerToLLVMIR(self.as_ref(), target_machine, ptr::null(), 0) }
        };
        if result.is_null() {
            Err(anyhow!("lowering to llvm failed"))
        } else {
            Ok(llvm::Module::new(result, target_machine))
        }
    }

    pub fn as_ref(&self) -> ModuleRef {
        unsafe { *self.module.as_ptr() }
    }
}
impl fmt::Debug for Module {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "MLIRModule({:p}, dialect = {:?})",
            self.as_ref(),
            self.dialect.borrow()
        )
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
        Self::new(self.as_ref(), self.dialect.borrow().clone())
    }
}

impl Emit for Module {
    const TYPE: OutputType = OutputType::EIRDialect;

    fn emit_output_type(&self) -> OutputType {
        match *self.dialect.borrow() {
            Dialect::EIR => OutputType::EIRDialect,
            Dialect::LLVM => OutputType::LLVMDialect,
            Dialect::Standard => OutputType::StandardDialect,
            _ => Self::TYPE,
        }
    }

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
    pub fn MLIRLowerModule(
        context: ContextRef,
        pass_manager: PassManagerRef,
        target_machine: TargetMachineRef,
        module: ModuleRef,
    ) -> ModuleRef;

    pub fn MLIRLowerToLLVMIR(
        module: ModuleRef,
        target_machine: TargetMachineRef,
        source_name: *const libc::c_char,
        source_name_len: libc::c_uint,
    ) -> llvm::ModuleRef;

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

    #[allow(unused)]
    pub fn MLIREmitToMemoryBuffer(M: ModuleRef) -> MemoryBufferRef;
}
