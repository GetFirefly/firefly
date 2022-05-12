mod mfa;

pub use self::mfa::ModuleFunctionArity;

use crate::term::Atom;

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
    pub module: Atom,
    /// Function name atom
    pub function: Atom,
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
    pub ptr: *const (),
}

/// Function symbols are read-only and pinned, and therefore Sync
unsafe impl Sync for FunctionSymbol {}

/// Function symbols are read-only and pinned, and therefore Send
unsafe impl Send for FunctionSymbol {}
