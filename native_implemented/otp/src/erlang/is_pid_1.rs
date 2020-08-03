#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(erlang:is_pid/1)]
pub fn result(term: Term) -> Term {
    term.is_pid().into()
}
