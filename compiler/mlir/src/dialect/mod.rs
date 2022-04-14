use std::fmt;

use crate::*;

pub mod cir;
//include!(concat!(env!("OUT_DIR"), "/dialects.rs"));

#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(C)]
pub enum DialectType {
    Other,
    None,
    CIR,
    Arithmetic,
    Func,
    ControlFlow,
    SCF,
    LLVM,
}
impl DialectType {
    pub fn namespace(&self) -> Option<&'static str> {
        match self {
            Self::Other | Self::None => None,
            Self::CIR => Some("cir"),
            Self::Arithmetic => Some("arith"),
            Self::Func => Some("func"),
            Self::ControlFlow => Some("cf"),
            Self::SCF => Some("scf"),
            Self::LLVM => Some("llvm"),
        }
    }
}
impl fmt::Display for DialectType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut name = format!("{:?}", self);
        name.make_ascii_lowercase();
        write!(f, "{}", &name)
    }
}

extern "C" {
    type MlirDialectImpl;
    type MlirDialectHandleImpl;
    type MlirDialectRegistryImpl;
}

#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Dialect(*mut MlirDialectImpl);
impl Dialect {
    #[inline]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    #[inline]
    pub fn context(self) -> Context {
        unsafe { mlir_dialect_get_context(self) }
    }

    #[inline]
    pub fn get_namespace(self) -> StringRef {
        unsafe { mlir_dialect_get_namespace(self) }
    }

    #[inline]
    pub fn type_id(self) -> TypeId {
        unsafe { mlir_dialect_get_type_id(self) }
    }
}
impl PartialEq for Dialect {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlir_dialect_equal(*self, *other) }
    }
}
impl fmt::Debug for Dialect {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let namespace = self.get_namespace();
        write!(f, "Dialect({})", &namespace)
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct DialectHandle(*mut MlirDialectHandleImpl);
impl DialectHandle {
    #[inline]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    pub fn get(ty: DialectType) -> Option<Self> {
        match ty {
            DialectType::Other | DialectType::None => None,
            DialectType::CIR => Some(unsafe { mlir_dialect_handle_get_cir() }),
            DialectType::Arithmetic => Some(unsafe { mlir_dialect_handle_get_arith() }),
            DialectType::Func => Some(unsafe { mlir_dialect_handle_get_func() }),
            DialectType::ControlFlow => Some(unsafe { mlir_dialect_handle_get_cf() }),
            DialectType::SCF => Some(unsafe { mlir_dialect_handle_get_scf() }),
            DialectType::LLVM => Some(unsafe { mlir_dialect_handle_get_llvm() }),
        }
    }

    #[inline(always)]
    pub fn get_namespace(self) -> StringRef {
        unsafe { mlir_dialect_handle_get_namespace(self) }
    }

    /// Registers this dialect in the given MLIR context
    #[inline(always)]
    pub fn register(self, context: Context) {
        unsafe {
            mlir_dialect_handle_register_dialect(self, context);
        }
    }

    /// Loads this dialect in the given MLIR context
    #[inline(always)]
    pub fn load(self, context: Context) {
        unsafe {
            mlir_dialect_handle_load_dialect(self, context);
        }
    }
}
impl fmt::Debug for DialectHandle {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let namespace = self.get_namespace();
        write!(f, "DialectHandle({})", &namespace)
    }
}

#[repr(transparent)]
pub struct DialectRegistry(*mut MlirDialectRegistryImpl);
impl DialectRegistry {
    #[inline(always)]
    pub fn new() -> Self {
        Self(unsafe { mlir_dialect_registry_create() })
    }

    #[inline(always)]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Registers the given Dialect with this registry
    #[inline(always)]
    pub fn register_dialect(&mut self, dialect: DialectHandle) {
        unsafe { mlir_dialect_handle_insert_dialect(dialect, self.0) }
    }

    /// Append the contents of the registry to the registry associated with the context.
    #[inline(always)]
    pub fn append_to_context(&mut self, context: Context) {
        unsafe { mlir_context_append_dialect_registry(context, self.0) }
    }
}
impl Drop for DialectRegistry {
    fn drop(&mut self) {
        unsafe { mlir_dialect_registry_destroy(self.0) }
    }
}

extern "C" {
    #[link_name = "mlirDialectRegistryCreate"]
    fn mlir_dialect_registry_create() -> *mut MlirDialectRegistryImpl;
    #[link_name = "mlirDialectRegistryDestroy"]
    fn mlir_dialect_registry_destroy(registry: *mut MlirDialectRegistryImpl);
    #[link_name = "mlirContextAppendDialectRegistry"]
    fn mlir_context_append_dialect_registry(
        context: Context,
        registry: *mut MlirDialectRegistryImpl,
    );
    #[link_name = "mlirDialectGetContext"]
    fn mlir_dialect_get_context(dialect: Dialect) -> Context;
    #[link_name = "mlirDialectHandleGetTypeID"]
    fn mlir_dialect_get_type_id(dialect: Dialect) -> TypeId;
    #[link_name = "mlirDialectEqual"]
    fn mlir_dialect_equal(a: Dialect, b: Dialect) -> bool;
    #[link_name = "mlirDialectGetNamespace"]
    fn mlir_dialect_get_namespace(dialect: Dialect) -> StringRef;
    #[link_name = "mlirGetDialectHandle__cir__"]
    fn mlir_dialect_handle_get_cir() -> DialectHandle;
    #[link_name = "mlirGetDialectHandle__arith__"]
    fn mlir_dialect_handle_get_arith() -> DialectHandle;
    #[link_name = "mlirGetDialectHandle__func__"]
    fn mlir_dialect_handle_get_func() -> DialectHandle;
    #[link_name = "mlirGetDialectHandle__cf__"]
    fn mlir_dialect_handle_get_cf() -> DialectHandle;
    #[link_name = "mlirGetDialectHandle__scf__"]
    fn mlir_dialect_handle_get_scf() -> DialectHandle;
    #[link_name = "mlirGetDialectHandle__llvm__"]
    fn mlir_dialect_handle_get_llvm() -> DialectHandle;
    #[link_name = "mlirDialectHandleGetNamespace"]
    fn mlir_dialect_handle_get_namespace(handle: DialectHandle) -> StringRef;
    #[link_name = "mlirDialectHandleLoadDialect"]
    fn mlir_dialect_handle_load_dialect(dialect: DialectHandle, context: Context);
    #[link_name = "mlirDialectHandleInsertDialect"]
    fn mlir_dialect_handle_insert_dialect(
        handle: DialectHandle,
        registry: *mut MlirDialectRegistryImpl,
    );
    #[link_name = "mlirDialectHandleRegisterDialect"]
    fn mlir_dialect_handle_register_dialect(dialect: DialectHandle, context: Context);
}
