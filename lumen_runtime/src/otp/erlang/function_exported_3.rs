//! ```elixir
//! @doc """
//! Returns `true` if the `function/arity` is exported from `module`.
//! """
//! @spec function_exported(module :: atom(), function :: atom(), arity :: 0..255)
//! ```
//!
//! Even functions that are defined and called from other modules in Lumen compiled code will not
//! show up as exported unless their `export` function is called such as
//!
//! ```
//! lumen_runtime::otp::erlang::self_0::export();
//! ```
//!
//! or directly registering the `code` function like
//!
//! ```
//! # use liblumen_alloc::erts::term::Atom;
//! let module = Atom::try_from_str("erlang").unwrap();
//! let function = Atom::try_from_str("self").unwrap();
//! let arity = 0;
//! let code = lumen_runtime::otp::erlang::self_0::code;
//!
//! lumen_runtime::code::export::insert(module, function, arity, code);
//! ```

#[cfg(test)]
mod test;

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::{Atom, Term};
use liblumen_alloc::Arity;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(function_exported/3)]
pub fn native(module: Term, function: Term, arity: Term) -> exception::Result {
    let module_atom: Atom = module.try_into()?;
    let function_atom: Atom = function.try_into()?;
    let arity_arity: Arity = arity.try_into()?;

    let exported =
        crate::code::export::contains_key(&module_atom, &function_atom, arity_arity).into();

    Ok(exported)
}
