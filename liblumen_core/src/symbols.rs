use core::ffi::c_void;

/// This struct represents the serialized form of a symbol table entry
///
/// This struct is intentionally laid out in memory to be identical to
/// `ModuleFunctionArity` with an extra field (the function pointer).
/// This allows the symbol table to use ModuleFunctionArity without
/// requiring
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FunctionSymbol {
    /// Module name atom
    pub module: *const u8,
    /// Function name atom
    pub function: *const u8,
    /// The arity of the function
    pub arity: u8,
    /// An opaque pointer to the function
    ///
    /// To call the function, it is necessary to transmute this
    /// pointer to one of the correct type. All Erlang functions
    /// expect terms, and return a term as result.
    ///
    /// NOTE: The target type must be marked `extern "C"`, in order
    /// to ensure that the correct calling convention is used.
    pub ptr: *const c_void,
}

// These are implemented for the compiler query system
// It is safe to do so, since the data is static and lives for the life of the program
unsafe impl Sync for FunctionSymbol {}
unsafe impl Send for FunctionSymbol {}
