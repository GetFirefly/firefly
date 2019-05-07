use super::*;

use std::sync::atomic::AtomicUsize;

use crate::exception::Result;
use crate::integer;
use crate::message::{self, Message};
use crate::otp::erlang;
use crate::process;
use crate::scheduler::{with_process, with_process_arc};

mod abs_1;
mod add_2;
mod and_2;
mod andalso_2;
mod append_element_2;
mod are_equal_after_conversion_2;
mod are_exactly_equal_2;
mod are_exactly_not_equal_2;
mod are_not_equal_after_conversion_2;
mod atom_to_binary_2;
mod atom_to_list_1;
mod band_2;
mod binary_part_2;
mod binary_part_3;
mod binary_to_atom_2;
mod binary_to_existing_atom_2;
mod binary_to_float_1;
mod binary_to_integer_1;
mod binary_to_integer_2;
mod binary_to_list_1;
mod binary_to_list_3;
mod binary_to_term_1;
mod binary_to_term_2;
mod bit_size_1;
mod bitstring_to_list_1;
mod bnot_1;
mod bor_2;
mod bsl_2;
mod bsr_2;
mod bxor_2;
mod byte_size_1;
mod cancel_timer_1;
mod cancel_timer_2;
mod ceil_1;
mod concatenate_2;
mod convert_time_unit_3;
mod delete_element_2;
mod div_2;
mod divide_2;
mod element_2;
mod error_1;
mod error_2;
mod exit_1;
mod hd_1;
mod insert_element_3;
mod is_alive_0;
mod is_atom_1;
mod is_binary_1;
mod is_bitstring_1;
mod is_boolean_1;
mod is_equal_or_less_than_2;
mod is_float_1;
mod is_greater_than_2;
mod is_greater_than_or_equal_2;
mod is_integer_1;
mod is_less_than_2;
mod is_list_1;
mod is_map_1;
mod is_map_key_2;
mod is_number_1;
mod is_pid_1;
mod is_record_2;
mod is_record_3;
mod is_reference_1;
mod is_tuple_1;
mod length_1;
mod list_to_atom_1;
mod list_to_binary_1;
mod list_to_bitstring_1;
mod list_to_existing_atom_1;
mod list_to_pid_1;
mod list_to_tuple_1;
mod make_ref_0;
mod map_get_2;
mod map_size_1;
mod max_2;
mod min_2;
mod monotonic_time_0;
mod monotonic_time_1;
mod multiply_2;
mod negate_1;
mod node_0;
mod not_1;
mod number_or_badarith_1;
mod or_2;
mod orelse_2;
mod raise_3;
mod read_timer_1;
mod register_2;
mod registered_0;
mod rem_2;
mod self_0;
mod send_2;
mod send_3;
mod setelement_3;
mod size_1;
mod split_binary_2;
mod start_timer_3;
mod start_timer_4;
mod subtract_2;
mod subtract_list_2;
mod throw_1;
mod tl_1;
mod tuple_size_1;
mod tuple_to_list_1;
mod unregister_1;
mod whereis_1;
mod xor_2;

enum FirstSecond {
    First,
    Second,
}

fn cancel_timer_message(timer_reference: Term, result: Term, process: &Process) -> Term {
    Term::slice_to_tuple(
        &[
            Term::str_to_atom("cancel_timer", DoNotCare).unwrap(),
            timer_reference,
            result,
        ],
        process,
    )
}

fn errors_badarg<F>(actual: F)
where
    F: FnOnce(&Process) -> Result,
{
    with_process(|process| assert_badarg!(actual(&process)))
}

fn errors_badarith<F>(actual: F)
where
    F: FnOnce(&Process) -> Result,
{
    with_process(|process| assert_badarith!(actual(&process)))
}

fn has_message(process: &Process, message: Term) -> bool {
    process
        .mailbox
        .lock()
        .unwrap()
        .iter()
        .any(|mailbox_message| match mailbox_message {
            Message::Process(process_message) => process_message == &message,
            Message::Heap(message::Heap {
                message: heap_message,
                ..
            }) => heap_message == &message,
        })
}

fn has_heap_message(process: &Process, message: Term) -> bool {
    process
        .mailbox
        .lock()
        .unwrap()
        .iter()
        .any(|mailbox_message| match mailbox_message {
            Message::Heap(message::Heap {
                message: heap_message,
                ..
            }) => heap_message == &message,
            _ => false,
        })
}

fn has_process_message(process: &Process, message: Term) -> bool {
    process
        .mailbox
        .lock()
        .unwrap()
        .iter()
        .any(|mailbox_message| match mailbox_message {
            Message::Process(process_message) => process_message == &message,
            _ => false,
        })
}

fn list_term(process: &Process) -> Term {
    let head_term = Term::str_to_atom("head", DoNotCare).unwrap();
    Term::cons(head_term, Term::EMPTY_LIST, process)
}

fn receive_message(process: &Process) -> Option<Term> {
    // always lock `heap` before `mailbox`
    let unlocked_heap = process.heap.lock().unwrap();
    let mut unlocked_mailbox = process.mailbox.lock().unwrap();

    unlocked_mailbox.receive(unlocked_heap)
}

static REGISTERED_NAME_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn registered_name() -> Term {
    Term::str_to_atom(
        format!(
            "registered{}",
            REGISTERED_NAME_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        )
        .as_ref(),
        DoNotCare,
    )
    .unwrap()
}

fn timeout_message(timer_reference: Term, message: Term, process: &Process) -> Term {
    Term::slice_to_tuple(
        &[
            Term::str_to_atom("timeout", DoNotCare).unwrap(),
            timer_reference,
            message,
        ],
        process,
    )
}
