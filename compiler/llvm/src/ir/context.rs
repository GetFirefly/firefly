use std::fmt;
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::path::Path;
use std::sync::Arc;

use anyhow::anyhow;
use paste::paste;

use firefly_session::Options;
use firefly_util::diagnostics::DiagnosticsHandler;

use super::*;
use crate::builder::OwnedBuilder;
use crate::diagnostics;
use crate::support::*;

extern "C" {
    type LlvmContext;
}

macro_rules! primitive_type_in_context {
    ($name:ident, $ty:ident, $mnemonic:ident) => {
        paste! {
            primitive_type_in_context!($name, $ty, $mnemonic, [<LLVM $name TypeInContext>]);
        }
    };
    ($name:ident, $ty:ident, $mnemonic:ident, $extern:ident) => {
        paste! {
            /// Gets a $name type in this context
            pub fn [<get_ $mnemonic _type>](self) -> $ty {
                extern "C" {
                    fn $extern(context: Context) -> $ty;
                }
                unsafe { $extern(self) }
            }
        }
    };
}

/// Represents a borrowed reference to an LLVM context
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Context(*const LlvmContext);
impl Context {
    /// Returns a reference to the global context
    pub fn global() -> Self {
        extern "C" {
            fn LLVMGetGlobalContext() -> Context;
        }
        unsafe { LLVMGetGlobalContext() }
    }

    /// Set the diagnostic handler of this context
    pub fn set_diagnostic_handler(
        self,
        handler: diagnostics::DiagnosticHandler,
        context: *mut std::ffi::c_void,
    ) {
        extern "C" {
            fn LLVMContextSetDiagnosticHandler(
                context: Context,
                handler: diagnostics::DiagnosticHandler,
                handler_context: *mut std::ffi::c_void,
            );
        }
        unsafe { LLVMContextSetDiagnosticHandler(self, handler, context) }
    }

    /// Get the diagnostic handler of this context.
    pub fn diagnostic_handler(self) -> diagnostics::DiagnosticHandler {
        extern "C" {
            fn LLVMContextGetDiagnosticHandler(context: Context) -> diagnostics::DiagnosticHandler;
        }
        unsafe { LLVMContextGetDiagnosticHandler(self) }
    }

    /// Get the handler context for the current diagnostic handler
    pub fn get_diagnostic_context(self) -> *mut std::ffi::c_void {
        extern "C" {
            fn LLVMContextGetDiagnosticContext(context: Context) -> *mut std::ffi::c_void;
        }
        unsafe { LLVMContextGetDiagnosticContext(self) }
    }

    primitive_type_in_context!(Void, VoidType, void);
    primitive_type_in_context!(Label, LabelType, label);
    primitive_type_in_context!(Token, TokenType, token);
    primitive_type_in_context!(Metadata, MetadataType, metadata);
    primitive_type_in_context!(Int1, IntegerType, i1);
    primitive_type_in_context!(Int8, IntegerType, i8);
    primitive_type_in_context!(Int16, IntegerType, i16);
    primitive_type_in_context!(Int32, IntegerType, i32);
    primitive_type_in_context!(Int64, IntegerType, i64);
    primitive_type_in_context!(Int128, IntegerType, i28);
    primitive_type_in_context!(Half, FloatType, f16);
    primitive_type_in_context!(BFloat, FloatType, bfloat);
    primitive_type_in_context!(Float, FloatType, f32);
    primitive_type_in_context!(Double, FloatType, f64);
    primitive_type_in_context!(X86FP80, FloatType, f80);
    primitive_type_in_context!(FP128, FloatType, f128);
    primitive_type_in_context!(PPCFP128, FloatType, ppc_f128);

    /// Gets an integer type with the given arbitrary bit width in this context
    pub fn get_integer_type(self, bitwidth: usize) -> IntegerType {
        extern "C" {
            fn LLVMIntTypeInContext(context: Context, bitwidth: u32) -> IntegerType;
        }
        unsafe { LLVMIntTypeInContext(self, bitwidth.try_into().unwrap()) }
    }

    /// Gets a struct type with the given fields
    ///
    /// The resulting struct type is _not_ packed, use `get_packed_struct_type` for that
    pub fn get_struct_type(self, elements: &[TypeBase]) -> StructType {
        self.build_struct_type(elements, false)
    }

    /// Gets a packed struct type with the given fields
    ///
    /// Use `get_struct_type` for unpacked structs
    pub fn get_packed_struct_type(self, elements: &[TypeBase]) -> StructType {
        self.build_struct_type(elements, true)
    }

    /// Creates a new named struct in this context with the given fields
    ///
    /// Set `packed` to true to create a packed struct.
    pub fn get_named_struct_type<S: Into<StringRef>>(
        self,
        name: S,
        elements: &[TypeBase],
        packed: bool,
    ) -> StructType {
        extern "C" {
            fn LLVMStructCreateNamed(
                context: Context,
                name: *const std::os::raw::c_char,
            ) -> StructType;
        }
        let name = name.into();
        let c_str = name.to_cstr();
        let ty = unsafe { LLVMStructCreateNamed(self, c_str.as_ptr()) };
        ty.set_body(elements, packed);
        ty
    }

    fn build_struct_type(self, elements: &[TypeBase], packed: bool) -> StructType {
        extern "C" {
            fn LLVMStructTypeInContext(
                context: Context,
                elements: *const TypeBase,
                num_elements: u32,
                packed: bool,
            ) -> StructType;
        }
        unsafe {
            LLVMStructTypeInContext(
                self,
                elements.as_ptr(),
                elements.len().try_into().unwrap(),
                packed,
            )
        }
    }

    /// Creates a constant value with string content in this context
    pub fn const_string<S: Into<StringRef>>(self, value: S) -> ConstantString {
        extern "C" {
            fn LLVMConstStringInContext(
                context: Context,
                ptr: *const u8,
                len: u32,
                disable_null_termination: bool,
            ) -> ConstantString;
        }
        let value = value.into();
        let disable_null_termination = value.is_null_terminated();
        unsafe {
            LLVMConstStringInContext(
                self,
                value.data,
                value.len.try_into().unwrap(),
                disable_null_termination,
            )
        }
    }

    pub fn const_struct(self, values: &[ConstantValue]) -> ConstantStruct {
        extern "C" {
            fn LLVMConstStructInContext(
                context: Context,
                values: *const ConstantValue,
                value_num: u32,
                packed: bool,
            ) -> ConstantStruct;
        }
        unsafe {
            LLVMConstStructInContext(
                self,
                values.as_ptr(),
                values.len().try_into().unwrap(),
                false,
            )
        }
    }

    /// Creates a new IR builder in this context
    #[inline]
    pub fn create_builder(self) -> OwnedBuilder {
        OwnedBuilder::new(self)
    }

    /// Creates a new module in this context
    pub fn create_module<S: Into<StringRef>>(self, name: S) -> OwnedModule {
        extern "C" {
            fn LLVMModuleCreateWithNameInContext(
                name: *const std::os::raw::c_char,
                context: Context,
            ) -> OwnedModule;
        }
        let name = name.into();
        let c_str = name.to_cstr();
        unsafe { LLVMModuleCreateWithNameInContext(c_str.as_ptr(), self) }
    }

    /// Creates a new basic block in this context without inserting it in a function
    pub fn create_block<S: Into<StringRef>>(self, name: S) -> Block {
        extern "C" {
            fn LLVMCreateBasicBlockInContext(
                context: Context,
                name: *const std::os::raw::c_char,
            ) -> Block;
        }
        let name = name.into();
        let c_str = name.to_cstr();
        unsafe { LLVMCreateBasicBlockInContext(self, c_str.as_ptr()) }
    }

    pub fn parse_string<I: AsRef<[u8]>>(self, input: I, name: &str) -> anyhow::Result<OwnedModule> {
        let buffer = MemoryBuffer::create_from_slice(input.as_ref(), name);
        self.parse_buffer(buffer)
    }

    pub fn parse_file<P: AsRef<Path>>(self, filename: P) -> anyhow::Result<OwnedModule> {
        let buffer = OwnedMemoryBuffer::create_from_file(filename)?;
        self.parse_buffer(buffer.borrow())
    }

    pub fn parse_buffer(self, buffer: MemoryBuffer<'_>) -> anyhow::Result<OwnedModule> {
        extern "C" {
            fn LLVMParseIRInContext(
                context: Context,
                buf: Buffer,
                module: *mut OwnedModule,
                error: *mut *mut std::os::raw::c_char,
            ) -> bool;
        }

        let mut module = MaybeUninit::uninit();
        let mut error = MaybeUninit::uninit();
        let failed =
            unsafe { LLVMParseIRInContext(self, *buffer, module.as_mut_ptr(), error.as_mut_ptr()) };

        if failed {
            let error = unsafe { OwnedStringRef::from_ptr(error.assume_init()) };
            Err(anyhow!("{}", &error))
        } else {
            Ok(unsafe { module.assume_init() })
        }
    }
}
impl Eq for Context {}
impl PartialEq for Context {
    fn eq(&self, other: &Self) -> bool {
        core::ptr::eq(self.0, other.0)
    }
}
impl fmt::Pointer for Context {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", self.0)
    }
}
impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", self.0)
    }
}

/// Represents an owned reference to an LLVM context
#[repr(transparent)]
pub struct OwnedContext(Context);
impl OwnedContext {
    /// Create a new owned context
    pub fn new(options: Arc<Options>, diagnostics: Arc<DiagnosticsHandler>) -> Self {
        extern "C" {
            fn LLVMContextCreate() -> Context;
        }
        let context = unsafe { LLVMContextCreate() };
        let handler_context = {
            let data = Box::new((
                context,
                Arc::downgrade(&options),
                Arc::downgrade(&diagnostics),
            ));
            Box::into_raw(data)
        };
        context.set_diagnostic_handler(diagnostics::diagnostic_handler, handler_context.cast());
        Self(context)
    }

    /// Returns a borrowed reference to this context
    pub fn borrow(&self) -> Context {
        self.0
    }
}
unsafe impl Send for OwnedContext {}
unsafe impl Sync for OwnedContext {}
impl Eq for OwnedContext {}
impl PartialEq for OwnedContext {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl fmt::Debug for OwnedContext {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", &self.0)
    }
}
impl Drop for OwnedContext {
    fn drop(&mut self) {
        extern "C" {
            fn LLVMContextDispose(context: Context);
        }
        unsafe {
            LLVMContextDispose(self.0);
        }
    }
}
impl Deref for OwnedContext {
    type Target = Context;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
