#[cfg(test)]
mod test;

use liblumen_alloc::erts::term::prelude::Term;

/// Distribution is not supported at this time.  Always returns `false`.
#[native_implemented::function(is_alive/0)]
pub fn result() -> Term {
    false.into()
}
