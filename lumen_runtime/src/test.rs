pub mod r#loop;

// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest,
// so disable property-based tests and associated helpers completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
pub mod proptest;
#[cfg(all(not(target_arch = "wasm32"), test))]
pub mod strategy;

#[cfg(all(not(target_arch = "wasm32"), test))]
pub use self::proptest::*;

use std::convert::TryInto;

use liblumen_alloc::atom;
use liblumen_alloc::erts::process::{Process, Status};
use liblumen_alloc::erts::term::prelude::*;

pub fn assert_exits<F: Fn(Option<Term>)>(
    process: &Process,
    expected_reason: Term,
    assert_stacktrace: F,
    source_substring: &str,
) {
    match *process.status.read() {
        Status::Exiting(ref runtime_exception) => {
            assert_eq!(runtime_exception.reason(), Some(expected_reason));
            assert_stacktrace(runtime_exception.stacktrace());

            let source_string = format!("{:?}", runtime_exception.source());

            assert!(
                source_string.contains(source_substring),
                "source ({}) does not contain `{}`",
                source_string,
                source_substring
            );
        }
        ref status => panic!("Process status ({:?}) is not exiting.", status),
    };
}

pub fn assert_exits_badarith(process: &Process, source_substring: &str) {
    assert_exits(process, atom!("badarith"), |_| {}, source_substring)
}

pub fn assert_exits_undef(
    process: &Process,
    module: Term,
    function: Term,
    arguments: Term,
    source_substring: &str,
) {
    assert_exits(
        process,
        atom!("undef"),
        |stacktrace| {
            let stacktrace_boxed_cons: Boxed<Cons> = stacktrace.unwrap().try_into().unwrap();
            let head = stacktrace_boxed_cons.head;

            assert_eq!(
                head,
                process
                    .tuple_from_slice(&[module, function, arguments, Term::NIL])
                    .unwrap()
            );
        },
        source_substring,
    );
}

pub fn badarity_reason(process: &Process, function: Term, args: Term) -> Term {
    let tag = atom!("badarity");
    let fun_args = process.tuple_from_slice(&[function, args]).unwrap();

    process.tuple_from_slice(&[tag, fun_args]).unwrap()
}
