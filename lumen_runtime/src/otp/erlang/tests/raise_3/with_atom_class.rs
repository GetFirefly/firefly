use super::*;

use crate::exception::Class::*;

mod error;
mod exit;
mod throw;

#[test]
fn with_other_class_errors_badarg() {
    let class_term = Term::str_to_atom("unsupported_class", DoNotCare).unwrap();
    let reason = Term::str_to_atom("reason", DoNotCare).unwrap();
    let stacktrace = Term::EMPTY_LIST;

    assert_badarg!(erlang::raise_3(class_term, reason, stacktrace));
}
