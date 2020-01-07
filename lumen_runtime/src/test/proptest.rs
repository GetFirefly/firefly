use proptest::test_runner::{Config, TestRunner};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use proptest::strategy::{Just, Strategy};
use proptest::test_runner::TestCaseResult;
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::atom;
use liblumen_alloc::erts::message::{self, Message};
use liblumen_alloc::erts::process::{Process, Status};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::{exception, Node};

use crate::process::spawn::options::Options;
use crate::scheduler::{Scheduler, Spawned};
use crate::test::r#loop;

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
    process.monitor_by_reference.lock().len()
}

pub fn monitored_count(process: &Process) -> usize {
    process.monitored_pid_by_reference.lock().len()
}

pub fn process(parent_process: &Process, options: Options) -> Spawned {
    let module = r#loop::module();
    let function = r#loop::function();
    let arguments = &[];
    let code = r#loop::code;

    Scheduler::spawn_code(parent_process, options, module, function, arguments, code).unwrap()
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
            "Child process did not exit.  Status is {:?}",
            status
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
        .run(
            &super::strategy::process().prop_flat_map(arc_process_fun),
            test,
        )
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

pub fn without_boolean_left_errors_badarg(
    source_file: &'static str,
    native: fn(Term, Term) -> exception::Result<Term>,
) {
    run(
        source_file,
        |arc_process| {
            (
                super::strategy::term::is_not_boolean(arc_process.clone()),
                super::strategy::term::is_boolean(),
            )
        },
        |(left_boolean, right_boolean)| {
            prop_assert_is_not_boolean!(native(left_boolean, right_boolean), left_boolean);

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
                super::strategy::term::is_not_function(arc_process.clone()),
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

pub fn without_number_errors_badarg(
    source_file: &'static str,
    native: fn(&Process, Term) -> exception::Result<Term>,
) {
    run(
        source_file,
        |arc_process| {
            (
                Just(arc_process.clone()),
                super::strategy::term::is_not_number(arc_process.clone()),
            )
        },
        |(arc_process, number)| {
            prop_assert_badarg!(
                native(&arc_process, number),
                format!("number ({}) is not an integer or float", number)
            );

            Ok(())
        },
    );
}

pub enum FirstSecond {
    First,
    Second,
}
