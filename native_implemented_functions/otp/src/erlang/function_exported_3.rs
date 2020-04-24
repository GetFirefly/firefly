//! ```elixir
//! @doc """
//! Returns `true` if the `function/arity` is exported from `module`.
//! """
//! @spec function_exported(module :: atom(), function :: atom(), arity :: 0..255)
//! ```

#[cfg(test)]
mod test;

use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::{Atom, Term};
use liblumen_alloc::{Arity, ModuleFunctionArity};

use liblumen_alloc::erts::apply::find_symbol;
use native_implemented_function::native_implemented_function;

#[native_implemented_function(function_exported/3)]
pub fn result(module: Term, function: Term, arity: Term) -> exception::Result<Term> {
    let module_atom: Atom = module.try_into().context("module must be an atom")?;
    let function_atom: Atom = function.try_into().context("function must be an atom")?;
    let arity_arity: Arity = arity.try_into().context("arity must be in 0-255")?;
    let module_function_arity = ModuleFunctionArity {
        module: module_atom,
        function: function_atom,
        arity: arity_arity,
    };

    Ok(find_symbol(&module_function_arity).is_some().into())
}
