#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use firefly_rt::term::Term;

#[native_implemented::function(erlang:is_tuple/1)]
pub fn result(term: Term) -> Term {
    term.is_tuple().into()
}
