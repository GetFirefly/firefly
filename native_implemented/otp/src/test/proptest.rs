pub mod float_to_string;

use std::convert::TryInto;

use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use num_bigint::BigInt;

use num_traits::Num;

use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestCaseResult, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::message::{self, Message};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::{exception, Node};

use crate::test::strategy::term::binary;
use crate::test::strategy::term::binary::sub::{bit_offset, byte_count, byte_offset};

use super::strategy;

pub fn arc_process_subbinary_to_arc_process_subbinary_two_less_than_length_start(
    (arc_process, binary): (Arc<Process>, Term),
) -> (
    impl Strategy<Value = Arc<Process>>,
    impl Strategy<Value = Term>,
    impl Strategy<Value = usize>,
) {
    let subbinary: Boxed<SubBinary> = binary.try_into().unwrap();
    let byte_count = subbinary.full_byte_len();

    // `start` must be 2 less than `byte_count` so that `length` can be at least 1
    // and still get a full byte
    (
        Just(arc_process.clone()),
        Just(binary),
        (1..=(byte_count - 2)),
    )
}

pub fn arc_process_to_arc_process_subbinary_zero_start_byte_count_length(
    arc_process: Arc<Process>,
) -> impl Strategy<Value = (Arc<Process>, Term, Term, Term)> {
    (
        Just(arc_process.clone()),
        binary::sub::with_size_range(
            byte_offset(),
            bit_offset(),
            byte_count(),
            (1_u8..=7_u8).boxed(),
            arc_process.clone(),
        ),
    )
        .prop_map(|(arc_process, binary)| {
            let subbinary: Boxed<SubBinary> = binary.try_into().unwrap();

            (
                arc_process.clone(),
                binary,
                arc_process.integer(0),
                arc_process.integer(subbinary.full_byte_len()),
            )
        })
}

pub fn external_arc_node() -> Arc<Node> {
    Arc::new(Node::new(
        1,
        Atom::try_from_str("node@external").unwrap(),
        0,
    ))
}

pub fn has_message(process: &Process, data: Term) -> bool {
    process.mailbox.lock().borrow().iter().any(|message| {
        &data
            == match message {
                Message::Process(message::Process { data }) => data,
                Message::HeapFragment(message::HeapFragment { data, .. }) => data,
            }
    })
}

pub fn has_heap_message(process: &Process, data: Term) -> bool {
    process
        .mailbox
        .lock()
        .borrow()
        .iter()
        .any(|message| match message {
            Message::HeapFragment(message::HeapFragment {
                data: message_data, ..
            }) => message_data == &data,
            _ => false,
        })
}

pub fn has_process_message(process: &Process, data: Term) -> bool {
    process
        .mailbox
        .lock()
        .borrow()
        .iter()
        .any(|message| match message {
            Message::Process(message::Process {
                data: message_data, ..
            }) => message_data == &data,
            _ => false,
        })
}

pub fn monitor_count(process: &Process) -> usize {
    process.monitor_by_reference.len()
}

pub fn monitored_count(process: &Process) -> usize {
    process.monitored_pid_by_reference.len()
}

pub fn number_to_integer_with_float(
    source_file: &'static str,
    result: fn(&Process, Term) -> exception::Result<Term>,
    non_zero_assertion: fn(Term, f64, Term) -> TestCaseResult,
) {
    run(
        source_file,
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::float(arc_process.clone()),
            )
        },
        |(arc_process, number)| {
            let result = result(&arc_process, number);

            prop_assert!(result.is_ok());

            let result_term = result.unwrap();

            prop_assert!(result_term.is_integer());

            let number_float: Float = number.try_into().unwrap();
            let number_f64: f64 = number_float.into();

            if number_f64.fract() == 0.0 {
                // f64::to_string() has no decimal point when there is no `fract`.
                let number_big_int =
                    <BigInt as Num>::from_str_radix(&number_f64.to_string(), 10).unwrap();
                let result_big_int: BigInt = result_term.try_into().unwrap();

                prop_assert_eq!(number_big_int, result_big_int);

                Ok(())
            } else {
                non_zero_assertion(number, number_f64, result_term)
            }
        },
    );
}

pub fn receive_message(process: &Process) -> Option<Term> {
    process
        .mailbox
        .lock()
        .borrow_mut()
        .receive(process)
        .map(|result| result.unwrap())
}

static REGISTERED_NAME_COUNTER: AtomicUsize = AtomicUsize::new(0);

pub fn registered_name() -> Term {
    Atom::str_to_term(
        format!(
            "registered{}",
            REGISTERED_NAME_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        )
        .as_str(),
    )
}

pub fn run<S: Strategy, F: Fn(Arc<Process>) -> S>(
    source_file: &'static str,
    arc_process_fun: F,
    test: impl Fn(S::Value) -> TestCaseResult,
) {
    TestRunner::new(Config::with_source_file(source_file))
        .run(&strategy::process().prop_flat_map(arc_process_fun), test)
        .unwrap();
}

pub fn timeout_message(timer_reference: Term, message: Term, process: &Process) -> Term {
    timer_message("timeout", timer_reference, message, process)
}

pub fn timer_message(tag: &str, timer_reference: Term, message: Term, process: &Process) -> Term {
    process.tuple_from_slice(&[Atom::str_to_term(tag), timer_reference, message])
}

pub fn with_binary_without_atom_encoding_errors_badarg(
    source_file: &'static str,
    result: fn(Term, Term) -> exception::Result<Term>,
) {
    run(
        source_file,
        |arc_process| {
            (
                strategy::term::is_binary(arc_process.clone()),
                strategy::term::is_not_atom(arc_process),
            )
        },
        |(binary, encoding)| {
            prop_assert_badarg!(
                result(binary, encoding),
                format!("invalid encoding name value: `{}` is not an atom", encoding)
            );

            Ok(())
        },
    );
}

pub fn with_binary_with_atom_without_name_encoding_errors_badarg(
    source_file: &'static str,
    result: fn(Term, Term) -> exception::Result<Term>,
) {
    run(
        source_file,
        |arc_process| {
            (
                strategy::term::is_binary(arc_process.clone()),
                strategy::term::atom::is_not_encoding(),
            )
        },
        |(binary, encoding)| {
            let encoding_atom: Atom = encoding.try_into().unwrap();

            prop_assert_badarg!(
                        result(binary, encoding),
                        format!("invalid atom encoding name: '{0}' is not one of the supported values (latin1, unicode, or utf8)", encoding_atom.name())
                    );

            Ok(())
        },
    );
}

pub fn with_integer_returns_integer(
    source_file: &'static str,
    result: fn(&Process, Term) -> exception::Result<Term>,
) {
    run(
        source_file,
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
            )
        },
        |(arc_process, number)| {
            prop_assert_eq!(result(&arc_process, number), Ok(number));

            Ok(())
        },
    );
}

pub fn without_boolean_left_errors_badarg(
    source_file: &'static str,
    result: fn(Term, Term) -> exception::Result<Term>,
) {
    run(
        source_file,
        |arc_process| {
            (
                strategy::term::is_not_boolean(arc_process.clone()),
                strategy::term::is_boolean(),
            )
        },
        |(left_boolean, right_boolean)| {
            prop_assert_is_not_boolean!(result(left_boolean, right_boolean), left_boolean);

            Ok(())
        },
    );
}

pub fn without_binary_errors_badarg(
    source_file: &'static str,
    result: fn(&Process, Term) -> exception::Result<Term>,
) {
    run(
        source_file,
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_binary(arc_process.clone()),
            )
        },
        |(arc_process, binary)| {
            prop_assert_badarg!(
                result(&arc_process, binary),
                format!("binary ({}) must be a binary", binary)
            );

            Ok(())
        },
    );
}

pub fn without_binary_with_encoding_is_not_binary(
    source_file: &'static str,
    result: fn(Term, Term) -> exception::Result<Term>,
) {
    run(
        source_file,
        |arc_process| {
            (
                strategy::term::is_not_binary(arc_process.clone()),
                strategy::term::is_encoding(),
            )
        },
        |(binary, encoding)| {
            prop_assert_is_not_binary!(result(binary, encoding), binary);

            Ok(())
        },
    );
}

pub fn without_bitstring_errors_badarg(
    source_file: &'static str,
    result: fn(&Process, Term) -> exception::Result<Term>,
) {
    run(
        source_file,
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_bitstring(arc_process.clone()),
            )
        },
        |(arc_process, bitstring)| {
            prop_assert_badarg!(
                result(&arc_process, bitstring),
                format!("bitstring ({}) is not a bitstring", bitstring)
            );

            Ok(())
        },
    );
}

pub fn without_float_errors_badarg(
    source_file: &'static str,
    result: fn(&Process, Term) -> exception::Result<Term>,
) {
    run(
        source_file,
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_float(arc_process.clone()),
            )
        },
        |(arc_process, float)| {
            prop_assert_badarg!(
                result(&arc_process, float),
                format!("float ({}) is not a float", float)
            );

            Ok(())
        },
    );
}

pub fn without_float_with_empty_options_errors_badarg(
    source_file: &'static str,
    result: fn(&Process, Term, Term) -> exception::Result<Term>,
) {
    run(
        source_file,
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_float(arc_process.clone()),
            )
        },
        |(arc_process, float)| {
            let options = Term::NIL;

            prop_assert_badarg!(
                result(&arc_process, float, options),
                format!("float ({}) is not a float", float)
            );

            Ok(())
        },
    );
}

pub fn without_function_errors_badarg(
    source_file: &'static str,
    result: fn(&Process, Term) -> exception::Result<Term>,
) {
    run(
        source_file,
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_function(arc_process.clone()),
            )
        },
        |(arc_process, function)| {
            prop_assert_badarg!(
                result(&arc_process, function),
                format!("function ({}) is not a function", function)
            );

            Ok(())
        },
    );
}

pub fn without_integer_errors_badarg(
    source_file: &'static str,
    result: fn(&Process, Term) -> exception::Result<Term>,
) {
    run(
        source_file,
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_integer(arc_process.clone()),
            )
        },
        |(arc_process, integer)| {
            prop_assert_badarg!(
                result(&arc_process, integer),
                format!("integer ({}) is not an integer", integer)
            );

            Ok(())
        },
    );
}

pub fn without_integer_integer_with_base_errors_badarg(
    source_file: &'static str,
    result: fn(&Process, Term, Term) -> exception::Result<Term>,
) {
    run(
        source_file,
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_integer(arc_process.clone()),
                strategy::term::is_base(arc_process.clone()),
            )
        },
        |(arc_process, integer, base)| {
            prop_assert_badarg!(
                result(&arc_process, integer, base),
                format!("integer ({}) is not an integer", integer)
            );

            Ok(())
        },
    );
}

pub fn with_integer_integer_without_base_base_errors_badarg(
    source_file: &'static str,
    result: fn(&Process, Term, Term) -> exception::Result<Term>,
) {
    run(
        source_file,
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
                strategy::term::is_not_base(arc_process.clone()),
            )
        },
        |(arc_process, integer, base)| {
            prop_assert_badarg!(
                result(&arc_process, integer, base),
                "base must be an integer in 2-36"
            );

            Ok(())
        },
    );
}

pub fn without_integer_dividend_errors_badarith(
    source_file: &'static str,
    result: fn(&Process, Term, Term) -> exception::Result<Term>,
) {
    run(
        source_file,
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_integer(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
            )
        },
        |(arc_process, dividend, divisor)| {
            prop_assert_badarith!(
                result(&arc_process, dividend, divisor),
                format!(
                    "dividend ({}) and divisor ({}) are not both numbers",
                    dividend, divisor
                )
            );

            Ok(())
        },
    );
}

pub fn with_integer_dividend_without_integer_divisor_errors_badarith(
    source_file: &'static str,
    result: fn(&Process, Term, Term) -> exception::Result<Term>,
) {
    run(
        source_file,
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
                strategy::term::is_not_integer(arc_process.clone()),
            )
        },
        |(arc_process, dividend, divisor)| {
            prop_assert_badarith!(
                result(&arc_process, dividend, divisor),
                format!(
                    "dividend ({}) and divisor ({}) are not both numbers",
                    dividend, divisor
                )
            );

            Ok(())
        },
    );
}

pub fn with_integer_dividend_with_zero_divisor_errors_badarith(
    source_file: &'static str,
    result: fn(&Process, Term, Term) -> exception::Result<Term>,
) {
    run(
        source_file,
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
                Just(arc_process.integer(0)),
            )
        },
        |(arc_process, dividend, divisor)| {
            prop_assert_badarith!(
                result(&arc_process, dividend, divisor),
                format!("divisor ({}) cannot be zero", divisor)
            );

            Ok(())
        },
    );
}

pub fn without_number_errors_badarg(
    source_file: &'static str,
    result: fn(&Process, Term) -> exception::Result<Term>,
) {
    run(
        source_file,
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_number(arc_process.clone()),
            )
        },
        |(arc_process, number)| {
            prop_assert_is_not_number!(result(&arc_process, number), number);

            Ok(())
        },
    );
}

pub fn without_timer_reference_errors_badarg(
    source_file: &'static str,
    result: fn(&Process, Term) -> exception::Result<Term>,
) {
    run(
        source_file,
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_reference(arc_process.clone()),
            )
        },
        |(arc_process, timer_reference)| {
            prop_assert_badarg!(
                result(&arc_process, timer_reference,),
                format!(
                    "timer_reference ({}) is not a local reference",
                    timer_reference
                )
            );

            Ok(())
        },
    );
}

pub enum FirstSecond {
    First,
    Second,
}
