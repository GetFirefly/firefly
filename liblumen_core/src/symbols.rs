use core::ffi::c_void;
#[cfg(all(unix, target_arch = "x86_64"))]
use core::mem;

#[cfg(all(unix, target_arch = "x86_64"))]
use crate::sys::dynamic_call::{self, DynamicCallee};

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
impl FunctionSymbol {
    /// Invokes the function bound to this symbol using the provided arguments
    ///
    /// It is crucially important to READ the following:
    ///
    /// There is absolutely no way to verify anything about the callee here,
    /// other than the function pointer itself is non-null; otherwise it is
    /// safeties off. You should never be creating `FunctionSymbol` by hand
    /// and calling them via `invoke` - it's completely unnecessary. Instead,
    /// this function is meant to support the ability of the scheduler/runtime
    /// to lookup codegen'd functions dynamically at runtime and invoke them,
    /// which is needed to spawn processes.
    ///
    /// This function assumes that all callees fit the following definition:
    ///
    ///   - They use the C calling convention
    ///   - They return a Term/usize value
    ///   - They accept zero or more Term/usize values
    ///
    /// Should any of these rules fail to be followed, who knows what kind of
    /// madness will ensue - ideally things explode immediately, but more likely
    /// is that things get bonkers, real quick.
    ///
    /// In the future, we may actually improve this to store the calling convention
    /// of the callee alongside the symbol, so that we can support a wider array
    /// of target functions; likely to make an interactive shell usable we'd need
    /// something like that. That would also significantly complicate the code though,
    /// so for the time being it is more useful to restrict the possible targets.
    ///
    /// The use of `usize` in the arguments/return value here is due to the lack
    /// of a `Term` definition in this crate - but Term is always convertible to
    /// `usize`, so it shouldn't be an issue in practice.
    #[cfg(all(unix, target_arch = "x86_64"))]
    pub unsafe fn invoke(&self, args: &[usize]) -> usize {
        let arity = self.arity;
        let num_args = args.len();
        debug_assert_eq!(arity as usize, num_args, "mismatched arity!");
        let args_ptr = args.as_ptr();

        let f = mem::transmute::<*const c_void, DynamicCallee>(self.ptr);

        if arity == 0 {
            return f();
        }

        dynamic_call::apply(f, args_ptr, num_args)
    }
}

// These are implemented for the compiler query system
// It is safe to do so, since the data is static and lives for the life of the program
unsafe impl Sync for FunctionSymbol {}
unsafe impl Send for FunctionSymbol {}
