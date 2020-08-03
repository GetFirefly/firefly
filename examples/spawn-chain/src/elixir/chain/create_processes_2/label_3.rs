//! ```elixir
//! # pushed to stack: (final_answer)
//! # returned from call: :ok
//! # full stack: (:ok, final_answer)
//! # returns: final_answer
//! final_answer
//! ```

use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::label]
fn result(ok: Term, final_answer: Term) -> Term {
    assert_eq!(ok, Atom::str_to_term("ok"));
    assert!(final_answer.is_integer());

    final_answer
}
