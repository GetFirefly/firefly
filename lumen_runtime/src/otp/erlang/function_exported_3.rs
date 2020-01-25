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
//! # use liblumen_alloc::erts::term::closure::Definition;
//! # use liblumen_alloc::erts::term::prelude::Atom;
//! let module = Atom::try_from_str("erlang").unwrap();
//! let function = Atom::try_from_str("self").unwrap();
//! let definition = Definition::Export { function };
//! let arity = 0;
//! let located_code = lumen_runtime::otp::erlang::self_0::LOCATED_CODE;
//!
//! lumen_runtime::code::insert(module, definition, arity, located_code);
//! ```

#[cfg(test)]
mod test;

use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::closure::Definition;
use liblumen_alloc::erts::term::prelude::{Atom, Term};
use liblumen_alloc::Arity;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(function_exported/3)]
pub fn native(module: Term, function: Term, arity: Term) -> exception::Result<Term> {
    let module_atom: Atom = module.try_into().context("module must be an atom")?;
    let function_atom: Atom = function.try_into().context("function must be an atom")?;
    let arity_arity: Arity = arity.try_into().context("arity must be in 0-255")?;

    let definition = Definition::Export {
        function: function_atom,
    };

    let exported = crate::code::contains_key(&module_atom, &definition, arity_arity).into();

    Ok(exported)
}
