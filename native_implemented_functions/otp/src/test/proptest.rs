pub mod float_to_string;

use std::convert::TryInto;

use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use num_bigint::BigInt;

use num_traits::Num;

use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestCaseResult, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::atom;
use liblumen_alloc::erts::message::{self, Message};
use liblumen_alloc::erts::process::{Process, Status};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::{exception, Node};

use crate::runtime::process::spawn::options::Options;
use crate::runtime::scheduler::{self, Spawned};

use crate::test::r#loop;
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
                arc_process.integer(0).unwrap(),
                arc_process.integer(subbinary.full_byte_len()).unwrap(),
            )
        })
}

pub fn cancel_timer_message(timer_reference: Term, result: Term, process: &Process) -> Term {
    timer_message("cancel_timer", timer_reference, result, process)
}

pub fn count_ones(term: Term) -> u32 {
    match term.decode().unwrap() {
        TypedTerm::SmallInteger(n) => n.count_ones(),
        TypedTerm::BigInteger(n) => n.as_ref().count_ones(),
        _ => panic!("Can't count 1s in non-integer"),
    }
}

pub fn external_arc_node() -> Arc<Node> {
    Arc::new(Node::new(
        1,
        Atom::try_from_str("node@external").unwrap(),
        0,
    ))
}

pub fn has_no_message(process: &Process) -> bool {
    process.mailbox.lock().borrow().len() == 0
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
    native: fn(&Process, Term) -> exception::Result<Term>,
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
            let result = native(&arc_process, number);

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

pub fn process(parent_process: &Process, options: Options) -> Spawned {
    let module = r#loop::module();
    let function = r#loop::function();
    let arguments = &[];
    let code = r#loop::code;

    scheduler::spawn_code(parent_process, options, module, function, arguments, code).unwrap()
}

pub fn prop_assert_exits<
    F: Fn(Option<Term>) -> proptest::test_runner::TestCaseResult,
    S: AsRef<str>,
>(
    process: &Process,
    expected_reason: Term,
    prop_assert_stacktrace: F,
    source_substring: S,
) -> proptest::test_runner::TestCaseResult {
    match *process.status.read() {
        Status::Exiting(ref runtime_exception) => {
            prop_assert_eq!(runtime_exception.reason(), Some(expected_reason));
            prop_assert_stacktrace(runtime_exception.stacktrace())?;

            let source_string = format!("{:?}", runtime_exception.source());

            let source_substring: &str = source_substring.as_ref();

            prop_assert!(
                source_string.contains(source_substring),
                "source ({}) does not contain `{}`",
                source_string,
                source_substring
            );

            Ok(())
        }
        ref status => Err(proptest::test_runner::TestCaseError::fail(format!(
            "Child process did not exit.  Status is {:?}. Scheduler is {:?}",
            status,
            scheduler::current()
        ))),
    }
}

pub fn prop_assert_exits_badarity<S: AsRef<str>>(
    process: &Process,
    fun: Term,
    args: Term,
    source_substring: S,
) -> proptest::test_runner::TestCaseResult {
    let tag = atom!("badarity");
    let fun_args = process.tuple_from_slice(&[fun, args]).unwrap();
    let reason = process.tuple_from_slice(&[tag, fun_args]).unwrap();

    prop_assert_exits(process, reason, |_| Ok(()), source_substring)
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
    process
        .tuple_from_slice(&[Atom::str_to_term(tag), timer_reference, message])
        .unwrap()
}

pub fn total_byte_len(term: Term) -> usize {
    match term.decode().unwrap() {
        TypedTerm::HeapBinary(heap_binary) => heap_binary.total_byte_len(),
        TypedTerm::SubBinary(subbinary) => subbinary.total_byte_len(),
        typed_term => panic!("{:?} does not have a total_byte_len", typed_term),
    }
}

pub fn with_binary_without_atom_encoding_errors_badarg(
    source_file: &'static str,
    native: fn(Term, Term) -> exception::Result<Term>,
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
                native(binary, encoding),
                format!("invalid encoding name value: `{}` is not an atom", encoding)
            );

            Ok(())
        },
    );
}

pub fn with_binary_with_atom_without_name_encoding_errors_badarg(
    source_file: &'static str,
    native: fn(Term, Term) -> exception::Result<Term>,
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
                        native(binary, encoding),
                        format!("invalid atom encoding name: '{0}' is not one of the supported values (latin1, unicode, or utf8)", encoding_atom.name())
                    );

            Ok(())
        },
    );
}

pub fn with_integer_returns_integer(
    source_file: &'static str,
    native: fn(&Process, Term) -> exception::Result<Term>,
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
            prop_assert_eq!(native(&arc_process, number), Ok(number));

            Ok(())
        },
    );
}

pub fn with_integer_left_without_integer_right_errors_badarith(
    source_file: &'static str,
    native: fn(&Process, Term, Term) -> exception::Result<Term>,
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
        |(arc_process, left, right)| {
            prop_assert_badarith!(
                native(&arc_process, left, right),
                format!(
                    "left_integer ({}) and right_integer ({}) are not both integers",
                    left, right
                )
            );

            Ok(())
        },
    );
}

pub fn with_positive_start_and_positive_length_returns_subbinary(
    source_file: &'static str,
    returns_subbinary: fn((Arc<Process>, Term, Term, Term)) -> TestCaseResult,
) {
    run(
        source_file,
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_binary::with_byte_len_range((3..=6).into(), arc_process.clone()),
            )
                .prop_flat_map(|(arc_process, binary)| {
                    let byte_len = total_byte_len(binary);

                    // `start` must be 2 less than `byte_len` so that `length` can be at least 1
                    (
                        Just(arc_process.clone()),
                        Just(binary),
                        (1..=(byte_len - 2)),
                    )
                })
                .prop_flat_map(|(arc_process, binary, start)| {
                    (
                        Just(arc_process.clone()),
                        Just(binary),
                        Just(start),
                        1..=(total_byte_len(binary) - start),
                    )
                })
                .prop_map(|(arc_process, binary, start, length)| {
                    (
                        arc_process.clone(),
                        binary,
                        arc_process.integer(start).unwrap(),
                        arc_process.integer(length).unwrap(),
                    )
                })
        },
        returns_subbinary,
    );
}

pub fn with_positive_start_and_negative_length_returns_subbinary(
    source_file: &'static str,
    returns_subbinary: fn((Arc<Process>, Term, Term, Term)) -> TestCaseResult,
) {
    run(
        source_file,
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_bitstring::with_byte_len_range(
                    (2..=4).into(),
                    arc_process.clone(),
                ),
            )
                .prop_flat_map(|(arc_process, binary)| {
                    let byte_len = total_byte_len(binary);

                    (Just(arc_process.clone()), Just(binary), (1..byte_len))
                })
                .prop_flat_map(|(arc_process, binary, start)| {
                    (
                        Just(arc_process.clone()),
                        Just(binary),
                        Just(start),
                        (-(start as isize))..=(-1),
                    )
                })
                .prop_map(|(arc_process, binary, start, length)| {
                    (
                        arc_process.clone(),
                        binary,
                        arc_process.integer(start).unwrap(),
                        arc_process.integer(length).unwrap(),
                    )
                })
        },
        returns_subbinary,
    );
}

pub fn with_size_start_and_negative_size_length_returns_binary(
    source_file: &'static str,
    returns_binary: fn((Arc<Process>, Term, Term, Term)) -> TestCaseResult,
) {
    run(
        source_file,
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_binary::with_byte_len_range(1..=4, arc_process.clone()),
            )
                .prop_map(|(arc_process, binary)| {
                    let byte_len = total_byte_len(binary);

                    (
                        arc_process.clone(),
                        binary,
                        arc_process.integer(byte_len).unwrap(),
                        arc_process.integer(-(byte_len as isize)).unwrap(),
                    )
                })
        },
        returns_binary,
    );
}

pub fn with_zero_start_and_size_length_returns_binary(
    source_file: &'static str,
    returns_binary: fn((Arc<Process>, Term, Term, Term)) -> TestCaseResult,
) {
    run(
        source_file,
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_binary(arc_process.clone()),
            )
                .prop_map(|(arc_process, binary)| {
                    (
                        arc_process.clone(),
                        binary,
                        arc_process.integer(0).unwrap(),
                        arc_process.integer(total_byte_len(binary)).unwrap(),
                    )
                })
        },
        returns_binary,
    );
}

pub fn without_boolean_left_errors_badarg(
    source_file: &'static str,
    native: fn(Term, Term) -> exception::Result<Term>,
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
            prop_assert_is_not_boolean!(native(left_boolean, right_boolean), left_boolean);

            Ok(())
        },
    );
}

pub fn without_binary_errors_badarg(
    source_file: &'static str,
    native: fn(&Process, Term) -> exception::Result<Term>,
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
                native(&arc_process, binary),
                format!("binary ({}) must be a binary", binary)
            );

            Ok(())
        },
    );
}

pub fn without_binary_with_encoding_is_not_binary(
    source_file: &'static str,
    native: fn(Term, Term) -> exception::Result<Term>,
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
            prop_assert_is_not_binary!(native(binary, encoding), binary);

            Ok(())
        },
    );
}

pub fn without_bitstring_errors_badarg(
    source_file: &'static str,
    native: fn(&Process, Term) -> exception::Result<Term>,
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
                native(&arc_process, bitstring),
                format!("bitstring ({}) is not a bitstring", bitstring)
            );

            Ok(())
        },
    );
}

pub fn without_float_errors_badarg(
    source_file: &'static str,
    native: fn(&Process, Term) -> exception::Result<Term>,
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
                native(&arc_process, float),
                format!("float ({}) is not a float", float)
            );

            Ok(())
        },
    );
}

pub fn without_float_with_empty_options_errors_badarg(
    source_file: &'static str,
    native: fn(&Process, Term, Term) -> exception::Result<Term>,
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
                native(&arc_process, float, options),
                format!("float ({}) is not a float", float)
            );

            Ok(())
        },
    );
}

pub fn without_function_errors_badarg(
    source_file: &'static str,
    native: fn(&Process, Term) -> exception::Result<Term>,
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
                native(&arc_process, function),
                format!("function ({}) is not a function", function)
            );

            Ok(())
        },
    );
}

pub fn without_integer_errors_badarg(
    source_file: &'static str,
    native: fn(&Process, Term) -> exception::Result<Term>,
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
                native(&arc_process, integer),
                format!("integer ({}) is not an integer", integer)
            );

            Ok(())
        },
    );
}

pub fn without_integer_integer_with_base_errors_badarg(
    source_file: &'static str,
    native: fn(&Process, Term, Term) -> exception::Result<Term>,
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
                native(&arc_process, integer, base),
                format!("integer ({}) is not an integer", integer)
            );

            Ok(())
        },
    );
}

pub fn with_integer_integer_without_base_base_errors_badarg(
    source_file: &'static str,
    native: fn(&Process, Term, Term) -> exception::Result<Term>,
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
                native(&arc_process, integer, base),
                "base must be an integer in 2-36"
            );

            Ok(())
        },
    );
}

pub fn without_integer_dividend_errors_badarith(
    source_file: &'static str,
    native: fn(&Process, Term, Term) -> exception::Result<Term>,
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
                native(&arc_process, dividend, divisor),
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
    native: fn(&Process, Term, Term) -> exception::Result<Term>,
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
                native(&arc_process, dividend, divisor),
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
    native: fn(&Process, Term, Term) -> exception::Result<Term>,
) {
    run(
        source_file,
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
                Just(arc_process.integer(0).unwrap()),
            )
        },
        |(arc_process, dividend, divisor)| {
            prop_assert_badarith!(
                native(&arc_process, dividend, divisor),
                format!("divisor ({}) cannot be zero", divisor)
            );

            Ok(())
        },
    );
}

pub fn without_integer_left_errors_badarith(
    source_file: &'static str,
    native: fn(&Process, Term, Term) -> exception::Result<Term>,
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
        |(arc_process, left, right)| {
            prop_assert_badarith!(
                native(&arc_process, left, right),
                format!(
                    "left_integer ({}) and right_integer ({}) are not both integers",
                    left, right
                )
            );

            Ok(())
        },
    );
}

pub fn without_number_errors_badarg(
    source_file: &'static str,
    native: fn(&Process, Term) -> exception::Result<Term>,
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
            prop_assert_is_not_number!(native(&arc_process, number), number);

            Ok(())
        },
    );
}

pub fn without_timer_reference_errors_badarg(
    source_file: &'static str,
    native: fn(&Process, Term) -> exception::Result<Term>,
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
                native(&arc_process, timer_reference,),
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
