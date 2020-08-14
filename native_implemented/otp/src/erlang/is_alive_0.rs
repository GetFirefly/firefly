use liblumen_alloc::erts::term::prelude::Term;

/// Distribution is not supported at this time.  Always returns `false`.
#[native_implemented::function(erlang:is_alive/0)]
pub fn result() -> Term {
    false.into()
}
