use std::cell::RefCell;
use std::fmt;
use std::mem::MaybeUninit;
use std::os;

use anyhow::anyhow;

use liblumen_llvm as llvm;
use liblumen_llvm::target::TargetMachineRef;
use liblumen_llvm::utils::{LLVMString, MemoryBufferRef};
use liblumen_session::{Emit, OutputType};
use liblumen_util as util;

use crate::context::PassManagerRef;

mod ffi {
    use liblumen_compiler_macros::foreign_struct;

    #[foreign_struct]
    pub struct Module;

    #[repr(C)]
    pub struct ToMLIRResult {
        pub(super) module: ModuleRef,
        pub(super) success: bool,
    }

    #[repr(C)]
    pub struct ToLLVMIRResult {
        pub(super) module: *const std::ffi::c_void,
        pub(super) success: bool,
    }
    impl ToLLVMIRResult {
        pub(super) fn as_mlir(&self) -> ModuleRef {
            use std::ffi::c_void;
            use std::mem;

            unsafe { mem::transmute::<*const c_void, ModuleRef>(self.module) }
        }

        pub(super) fn as_llvm_ir(&self) -> super::llvm::ModuleRef {
            use super::llvm;
            use std::ffi::c_void;
            use std::mem;

            unsafe { mem::transmute::<*const c_void, llvm::ModuleRef>(self.module) }
        }
    }
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
    const DEFAULT_NAME: &'static str = "nofile";

    pub fn new(ptr: ModuleRef, dialect: Dialect) -> Self {
        assert!(!ptr.is_null());
        Self {
            module: RefCell::new(ptr),
            dialect: RefCell::new(dialect),
        }
    }

    pub fn is_valid(&self) -> bool {
        unsafe { MLIRVerifyModule(self.as_ref()) }
    }

    pub fn lower(&self, context: &Context) -> anyhow::Result<bool> {
        let pass_manager = context.pass_manager_ref();
        let result = unsafe { MLIRLowerModule(context.as_ref(), pass_manager, self.as_ref()) };
        let module = result.module;
        let success = result.success;
        if module.is_null() {
            return Err(anyhow!(
                "unknown error occurred during lowering to llvm dialect"
            ));
        }
        self.module.replace(module);
        if success {
            self.dialect.replace(Dialect::LLVM);
        } else {
            self.dialect.replace(Dialect::None);
        }
        return Ok(success);
    }

    pub fn lower_to_llvm_ir(
        &self,
        context: &Context,
        llvm_context: &llvm::Context,
        source_name: Option<String>,
    ) -> anyhow::Result<Result<llvm::module::Module, ()>> {
        let target_machine = context.target_machine_ref();

        let source_name_bytes = source_name
            .as_ref()
            .map(|s| s.as_bytes())
            .unwrap_or_else(|| Self::DEFAULT_NAME.as_bytes());
        let source_name_ptr = source_name_bytes.as_ptr();
        let source_name_len = source_name_bytes.len();

        let result = unsafe {
            MLIRLowerToLLVMIR(
                self.as_ref(),
                llvm_context.as_ref(),
                target_machine,
                source_name_ptr as *const libc::c_char,
                source_name_len as libc::c_uint,
            )
        };
        if result.module.is_null() {
            return Err(anyhow!("unknown error occurred during lowering to llvm ir"));
        }
        if result.success {
            Ok(Ok(llvm::Module::new(result.as_llvm_ir(), target_machine)))
        } else {
            self.module.replace(result.as_mlir());
            Ok(Err(()))
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
            Dialect::None => OutputType::MLIR,
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
    pub fn MLIRVerifyModule(module: ModuleRef) -> bool;

    pub fn MLIRLowerModule(
        context: ContextRef,
        pass_manager: PassManagerRef,
        module: ModuleRef,
    ) -> ffi::ToMLIRResult;

    pub fn MLIRLowerToLLVMIR(
        module: ModuleRef,
        context: llvm::ContextRef,
        target_machine: TargetMachineRef,
        source_name: *const libc::c_char,
        source_name_len: libc::c_uint,
    ) -> ffi::ToLLVMIRResult;

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
