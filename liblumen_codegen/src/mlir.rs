use std::ffi::CString;
use std::convert::{AsRef, AsMut};
use std::mem::MaybeUninit;
use std::path::Path;
use std::os;
use std::mem;
use std::ptr;
use std::fmt;

use anyhow::anyhow;

use libeir_ir as eir;

use liblumen_session::{Emit, Options, OutputType};
use liblumen_util as util;

use crate::llvm::{self, string::LLVMString};
use crate::ffi::{CodeGenOptLevel, CodeGenOptSize};
use super::{Result, CodegenError};

extern { pub type ContextImpl; }
extern { pub type DiagnosticEngine; }
extern { pub type DiagnosticInfo; }
extern { pub type Location; }
extern { pub type ModuleBuilder; }
extern { pub type ModuleImpl; }

/// TargetDialect
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub enum Dialect {
    #[allow(dead_code)]
    Other,
    None,
    EIR,
    Standard,
    LLVM,
}

pub struct Context<'a> {
    context: &'a mut ContextImpl,
}
impl Context<'static> {
    pub fn new() -> Self {
        let context = unsafe { MLIRCreateContext() };
        Self {
            context,
        }
    }
}
impl<'a> Context<'a> {
    pub fn new_module(&self, name: &str) -> &'static ModuleBuilder {
        let module_name = CString::new(name).unwrap();
        unsafe { MLIRCreateModuleBuilder(self.as_ref(), module_name.as_ptr()) }
    }

    pub fn parse_file<P: AsRef<Path>>(&mut self, filename: P) -> Result<Module<'a>> {
        let s = filename.as_ref().to_string_lossy().into_owned();
        let f = CString::new(s)?;
        let result = unsafe { MLIRParseFile(self.as_mut(), f.as_ptr()) };
        if result.is_null() {
            Err(anyhow::Error::new(CodegenError))
        } else {
            let m = unsafe { mem::transmute::<*mut ModuleImpl, &'a mut ModuleImpl>(result) };
            Ok(Module(m))
        }
    }
}
impl<'a> fmt::Debug for Context<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MLIRContext({:p})", self.context as *const _)
    }
}
impl<'a> AsRef<ContextImpl> for Context<'a> {
    fn as_ref(&self) -> &ContextImpl {
        self.context
    }
}
impl<'a> AsMut<ContextImpl> for Context<'a> {
    fn as_mut(&mut self) -> &mut ContextImpl {
        self.context
    }
}

#[repr(transparent)]
pub struct Module<'a>(&'a mut ModuleImpl);

impl<'a> Module<'a> {
    pub fn lower<'c>(&mut self, context: &mut Context<'c>, dialect: Dialect, opt: CodeGenOptLevel) -> Result<()> {
        let result = unsafe {
            MLIRLowerModule(context.as_mut(), self.as_mut(), dialect, opt)
        };
        if !result.is_null() {
            self.0 = unsafe { mem::transmute::<*mut ModuleImpl, &'a mut ModuleImpl>(result) };
            return Ok(());
        }
        let err = anyhow::Error::new(CodegenError)
            .context("lowering failed");
        Err(err)
    }

    pub fn lower_to_llvm_ir<'b>(
        &mut self,
        opt: CodeGenOptLevel,
        size: CodeGenOptSize,
        target_machine: &mut llvm::TargetMachine,
    ) -> Result<llvm::Module<'b>>
    {
        let result = unsafe {
            MLIRLowerToLLVMIR(
                self.as_mut(),
                opt,
                size,
                target_machine,
            )
        };
        if result.is_null() {
            Err(anyhow::Error::new(CodegenError))
        } else {
            let m = unsafe {
                mem::transmute::<*mut llvm::ModuleImpl, &'static mut llvm::ModuleImpl>(result)
            };
            Ok(llvm::Module::new(m, target_machine))
        }
    }
}
impl<'a> fmt::Debug for Module<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MLIRModule({:p})", self.0 as *const _)
    }
}
impl<'a> Eq for Module<'a> {}
impl<'a> PartialEq for Module<'a> {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.0 as *const _, other.0 as *const _)
    }
}
impl Clone for Module<'static> {
    fn clone(&self) -> Module<'static> {
        let ptr = self.0 as *const _ as *mut ModuleImpl;
        let m = unsafe {
            mem::transmute::<*mut ModuleImpl, &'static mut ModuleImpl>(ptr)
        };
        Self(m)
    }
}
impl<'a> AsRef<ModuleImpl> for Module<'a> {
    fn as_ref(&self) -> &ModuleImpl {
        self.0
    }
}
impl<'a> AsMut<ModuleImpl> for Module<'a> {
    fn as_mut(&mut self) -> &mut ModuleImpl {
        self.0
    }
}

impl<'a> Emit for Module<'a> {
    const TYPE: OutputType = OutputType::EIRDialect;

    fn emit(&self, f: &mut std::fs::File) -> anyhow::Result<()> {
        let fd = util::fs::get_file_descriptor(f);
        let mut err_string = MaybeUninit::uninit();
        let failed = unsafe {
            MLIREmitToFileDescriptor(self.as_ref(), fd, err_string.as_mut_ptr())
        };

        if failed {
            let err_string = LLVMString::new(unsafe { err_string.assume_init() });
            return Err(anyhow!("{}", err_string));
        }

        Ok(())
    }
}

extern "C" {
    pub fn MLIRCreateContext() -> &'static mut ContextImpl;

    pub fn MLIRCreateModuleBuilder(context: &ContextImpl, name: *const libc::c_char) -> &'static mut ModuleBuilder;

    pub fn MLIRCreateLocation(
        context: &ContextImpl,
        filename: *const libc::c_char,
        line: libc::c_uint,
        column: libc::c_uint
    ) -> &'static mut Location;

    pub fn MLIRParseFile(context: &mut ContextImpl, filename: *const libc::c_char) -> *mut ModuleImpl;

    pub fn MLIRLowerModule(
        context: &mut ContextImpl,
        module: &mut ModuleImpl,
        dialect: Dialect,
        opt: CodeGenOptLevel
    ) -> *mut ModuleImpl;

    pub fn MLIRLowerToLLVMIR(
        module: &mut ModuleImpl,
        opt: CodeGenOptLevel,
        size: CodeGenOptSize,
        target_machine: &mut llvm::TargetMachine
    ) -> *mut llvm::ModuleImpl;

    #[cfg(not(windows))]
    pub fn MLIREmitToFileDescriptor(
        M: &ModuleImpl,
        fd: os::unix::io::RawFd,
        error_message: *mut *mut libc::c_char
    ) -> bool;

    #[cfg(windows)]
    pub fn MLIREmitToFileDescriptor(
        M: &ModuleImpl,
        fd: os::windows::io::RawHandle,
        error_message: *mut *mut libc::c_char
    ) -> bool;

    pub fn MLIREmitToMemoryBuffer(M: &ModuleImpl) -> llvm::memory_buffer::MemoryBufferRef;
}

#[no_mangle]
pub unsafe extern "C" fn EIRSpanToMLIRLocation(_start: libc::c_uint, _end: libc::c_uint) -> &'static Location {
    unimplemented!()
}

pub fn generate_mlir(_options: &Options, _module: &eir::Module) -> Result<Module<'static>> {
    unimplemented!();
}
