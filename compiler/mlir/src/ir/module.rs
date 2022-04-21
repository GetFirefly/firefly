use std::fmt;
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::os;
use std::os::raw::c_char;

use anyhow::anyhow;

use liblumen_util as util;
use liblumen_util::emit::Emit;

use super::*;
use crate::support::{OwnedStringRef, StringRef};
use crate::Context;

extern "C" {
    type MlirModule;
}

/// This type represents a non-owned reference to an MLIR module
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Module(*mut MlirModule);
impl Module {
    #[inline(always)]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Returns the module name, if one was provided
    pub fn name(&self) -> Option<StringRef> {
        let name = unsafe { mlir_module_get_name(*self) };
        if name.is_null() {
            None
        } else {
            Some(name)
        }
    }

    /// Set the data layout string for this module
    pub fn set_data_layout<S: Into<StringRef>>(&self, layout: S) {
        self.set_attribute_by_name("llvm.data_layout", StringAttr::get(self.context(), layout));
    }

    /// Set the target triple for this module
    pub fn set_target_triple<S: Into<StringRef>>(&self, triple: S) {
        self.set_attribute_by_name(
            "llvm.target_triple",
            StringAttr::get(self.context(), triple),
        );
    }

    /// Returns the module body as a Block
    pub fn body(&self) -> Block {
        unsafe { mlir_module_get_body(*self) }
    }
}
impl Operation for Module {
    fn context(&self) -> Context {
        unsafe { mlir_module_get_context(*self) }
    }
    fn base(&self) -> OperationBase {
        unsafe { mlir_module_get_operation(*self) }
    }
}
impl TryFrom<OperationBase> for Module {
    type Error = InvalidTypeCastError;

    fn try_from(op: OperationBase) -> Result<Self, Self::Error> {
        if unsafe { mlir_operation_isa_module(op) } {
            Ok(unsafe { mlir_module_from_operation(op) })
        } else {
            Err(InvalidTypeCastError)
        }
    }
}
impl Into<OperationBase> for Module {
    #[inline]
    fn into(self) -> OperationBase {
        unsafe { mlir_module_get_operation(self) }
    }
}
impl Eq for Module {}
impl PartialEq for Module {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        core::ptr::eq(self.0, other.0)
    }
}
impl fmt::Pointer for Module {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", self.0)
    }
}
impl fmt::Debug for Module {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(name) = self.name() {
            write!(f, "Module({} @ {:p})", name, self)
        } else {
            write!(f, "Module(nofile @ {:p})", self)
        }
    }
}
impl fmt::Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let op: OperationBase = (*self).into();
        write!(f, "{}", &op)
    }
}
impl Emit for Module {
    fn file_type(&self) -> Option<&'static str> {
        Some("mlir")
    }

    fn emit(&self, f: &mut std::fs::File) -> anyhow::Result<()> {
        let fd = util::fs::get_file_descriptor(f);
        let mut error = MaybeUninit::uninit();
        let failed = unsafe { mlir_emit_to_file_descriptor(*self, fd, error.as_mut_ptr()) };

        if failed {
            let error = unsafe { OwnedStringRef::from_ptr(error.assume_init()) };
            Err(anyhow!("{}", &error))
        } else {
            Ok(())
        }
    }
}

/// This type represents an owned reference to an MLIR module
#[repr(transparent)]
pub struct OwnedModule(Module);
impl OwnedModule {
    /// Creates a new empty module with the given source location
    pub fn new(loc: Location) -> Self {
        unsafe { mlir_module_create_empty(loc) }
    }

    /// Parses an MLIR module from the given input string
    pub fn parse_string<S: Into<StringRef>>(context: Context, input: S) -> anyhow::Result<Self> {
        let base = unsafe { mlir_module_create_parse(context, input.into()) };
        if base.is_null() {
            Err(anyhow!("failed to parse module from the given string"))
        } else {
            Ok(Self(base))
        }
    }

    /// Parses an MLIR module from the file at the given path
    pub fn parse_file<S: Into<StringRef>>(context: Context, path: S) -> anyhow::Result<Self> {
        let base = unsafe { mlir_parse_file(context, path.into()) };
        if base.is_null() {
            Err(anyhow!("failed to parse module from the given string"))
        } else {
            Ok(Self(base))
        }
    }

    /// Returns the module name, if one was provided
    pub fn name(&self) -> Option<StringRef> {
        let name = unsafe { mlir_module_get_name(self.0) };
        if name.is_null() {
            None
        } else {
            Some(name)
        }
    }
}
unsafe impl Send for OwnedModule {}
unsafe impl Sync for OwnedModule {}
impl Clone for OwnedModule {
    fn clone(&self) -> Self {
        unsafe { mlir_module_clone(self.0) }
    }
}
impl Drop for OwnedModule {
    fn drop(&mut self) {
        unsafe { mlir_module_destroy(self.0) }
    }
}
impl Deref for OwnedModule {
    type Target = Module;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl AsRef<Module> for OwnedModule {
    fn as_ref(&self) -> &Module {
        &self.0
    }
}
impl Operation for OwnedModule {
    fn context(&self) -> Context {
        self.0.context()
    }
    fn base(&self) -> OperationBase {
        self.0.base()
    }
}
impl fmt::Debug for OwnedModule {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}
impl fmt::Display for OwnedModule {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl Eq for OwnedModule {}
impl PartialEq for OwnedModule {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Emit for OwnedModule {
    fn file_type(&self) -> Option<&'static str> {
        self.0.file_type()
    }

    fn emit(&self, f: &mut std::fs::File) -> anyhow::Result<()> {
        self.0.emit(f)
    }
}

extern "C" {
    #[link_name = "mlirOperationIsAModule"]
    fn mlir_operation_isa_module(op: OperationBase) -> bool;
    #[link_name = "mlirModuleCreateEmpty"]
    fn mlir_module_create_empty(loc: Location) -> OwnedModule;
    #[link_name = "mlirModuleCreateParse"]
    fn mlir_module_create_parse(context: Context, input: StringRef) -> Module;
    #[link_name = "mlirModuleGetContext"]
    fn mlir_module_get_context(module: Module) -> Context;
    #[link_name = "mlirModuleGetName"]
    fn mlir_module_get_name(module: Module) -> StringRef;
    #[link_name = "mlirModuleGetBody"]
    fn mlir_module_get_body(module: Module) -> Block;
    #[link_name = "mlirModuleClone"]
    fn mlir_module_clone(module: Module) -> OwnedModule;
    #[link_name = "mlirModuleDestroy"]
    fn mlir_module_destroy(module: Module);
    #[link_name = "mlirModuleGetOperation"]
    fn mlir_module_get_operation(module: Module) -> OperationBase;
    #[link_name = "mlirModuleFromOperation"]
    fn mlir_module_from_operation(op: OperationBase) -> Module;
    #[cfg(not(windows))]
    #[link_name = "MLIREmitToFileDescriptor"]
    fn mlir_emit_to_file_descriptor(
        m: Module,
        fd: os::unix::io::RawFd,
        error_message: *mut *mut c_char,
    ) -> bool;
    #[cfg(windows)]
    #[link_name = "MLIREmitToFileDescriptor"]
    fn mlir_emit_to_file_descriptor(
        m: Module,
        fd: os::windows::io::RawHandle,
        error_message: *mut *mut c_char,
    ) -> bool;
    #[link_name = "mlirParseFile"]
    fn mlir_parse_file(context: Context, filename: StringRef) -> Module;
}
