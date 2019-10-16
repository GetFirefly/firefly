pub mod r#loop;

// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
pub mod strategy;

use std::sync::atomic::AtomicUsize;

use num_bigint::BigInt;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::message::{self, Message};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::binary::MaybePartialByte;
use liblumen_alloc::erts::term::{atom_unchecked, BigInteger, Boxed, Term, TypedTerm};

use crate::process::spawn::options::Options;
use crate::scheduler::{with_process, Scheduler, Spawned};

pub fn cancel_timer_message(timer_reference: Term, result: Term, process: &Process) -> Term {
    timer_message("cancel_timer", timer_reference, result, process)
}

pub fn count_ones(term: Term) -> u32 {
    match term.to_typed_term().unwrap() {
        TypedTerm::SmallInteger(small_integer) => {
            let i: isize = small_integer.into();

            i.count_ones()
        }
        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
            TypedTerm::BigInteger(big_integer) => count_ones_in_big_integer(big_integer),
            _ => panic!("Can't count 1s in non-integer"),
        },
        _ => panic!("Can't count 1s in non-integer"),
    }
}

pub fn count_ones_in_big_integer(big_integer: Boxed<BigInteger>) -> u32 {
    let big_int: &BigInt = big_integer.as_ref().into();

    big_int
        .to_signed_bytes_be()
        .iter()
        .map(|b| b.count_ones())
        .sum()
}

pub fn errors_badarg<F>(actual: F)
where
    F: FnOnce(&Process) -> exception::Result,
{
    with_process(|process| assert_badarg!(actual(&process)))
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

pub fn list_term(process: &Process) -> Term {
    let head_term = atom_unchecked("head");

    process.cons(head_term, Term::NIL).unwrap()
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
    atom_unchecked(
        format!(
            "registered{}",
            REGISTERED_NAME_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        )
        .as_ref(),
    )
}

pub fn timeout_message(timer_reference: Term, message: Term, process: &Process) -> Term {
    timer_message("timeout", timer_reference, message, process)
}

pub fn timer_message(tag: &str, timer_reference: Term, message: Term, process: &Process) -> Term {
    process
        .tuple_from_slice(&[atom_unchecked(tag), timer_reference, message])
        .unwrap()
}

pub fn total_byte_len(term: Term) -> usize {
    match term.to_typed_term().unwrap() {
        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
            TypedTerm::HeapBinary(heap_binary) => heap_binary.total_byte_len(),
            TypedTerm::SubBinary(subbinary) => subbinary.total_byte_len(),
            unboxed_typed_term => panic!(
                "unboxed {:?} does not have a total_byte_len",
                unboxed_typed_term
            ),
        },
        typed_term => panic!("{:?} does not have a total_byte_len", typed_term),
    }
}

pub enum FirstSecond {
    First,
    Second,
}
