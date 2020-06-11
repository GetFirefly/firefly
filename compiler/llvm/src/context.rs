use std::fmt;
use std::mem::MaybeUninit;
use std::path::Path;
use std::ptr;
use std::sync::Arc;

use anyhow::anyhow;

use liblumen_util::diagnostics::DiagnosticsHandler;

use crate::module::{Module, ModuleImpl};
use crate::target::TargetMachineRef;
use crate::utils::{LLVMString, MemoryBuffer};
use crate::Result;

pub type ContextImpl = crate::sys::LLVMContext;
pub type ContextRef = crate::sys::prelude::LLVMContextRef;

#[repr(transparent)]
pub struct Context {
    context: ContextRef,
}
impl Context {
    pub fn new(diagnostics: Arc<DiagnosticsHandler>) -> Self {
        use crate::diagnostics;
        use crate::sys::core::LLVMContextSetDiagnosticHandler;

        let context = unsafe { crate::sys::core::LLVMContextCreate() };
        unsafe {
            let data = Box::new((context, Arc::downgrade(&diagnostics)));
            let data = Box::into_raw(data);
            LLVMContextSetDiagnosticHandler(
                context,
                Some(diagnostics::diagnostic_handler),
                data.cast(),
            );
        }
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
        use crate::sys::ir_reader::LLVMParseIRInContext;

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
//        unsafe { crate::sys::core::LLVMContextDispose(self.context); }
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
