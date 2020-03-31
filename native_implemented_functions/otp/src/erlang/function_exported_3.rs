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
//! liblumen_otp::erlang::self_0::export();
//! ```
//!
//! or directly registering the `code` function like
//!
//! ```
//! # use liblumen_alloc::erts::term::prelude::Atom;
//! let module = Atom::try_from_str("erlang").unwrap();
//! let function = Atom::try_from_str("self").unwrap();
//! let arity = 0;
//! let code = liblumen_otp::erlang::self_0::code;
//!
//! liblumen_otp::runtime::code::export::insert(module, function, arity, code);
//! ```

#[cfg(test)]
mod test;

use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::{Atom, Term};
use liblumen_alloc::Arity;

use native_implemented_function::native_implemented_function;

#[native_implemented_function(function_exported/3)]
pub fn native(module: Term, function: Term, arity: Term) -> exception::Result<Term> {
    let module_atom: Atom = module.try_into().context("module must be an atom")?;
    let function_atom: Atom = function.try_into().context("function must be an atom")?;
    let arity_arity: Arity = arity.try_into().context("arity must be in 0-255")?;

    let exported =
        crate::runtime::code::export::contains_key(&module_atom, &function_atom, arity_arity)
            .into();

    Ok(exported)
}
