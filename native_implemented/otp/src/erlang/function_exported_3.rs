//! ```elixir
//! @doc """
//! Returns `true` if the `function/arity` is exported from `module`.
//! """
//! @spec function_exported(module :: atom(), function :: atom(), arity :: 0..255)
//! ```

use std::convert::TryInto;
use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::function::{Arity, find_symbol, ModuleFunctionArity};
use firefly_rt::term::{Atom, Term};

#[native_implemented::function(erlang:function_exported/3)]
pub fn result(module: Term, function: Term, arity: Term) -> Result<Term, NonNull<ErlangException>> {
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
