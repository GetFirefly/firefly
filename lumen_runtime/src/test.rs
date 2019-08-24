pub mod r#loop;

// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
pub mod strategy;

use std::sync::atomic::AtomicUsize;

use liblumen_alloc::erts::message::{self, Message};
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::term::{atom_unchecked, Term};

pub fn has_no_message(process: &ProcessControlBlock) -> bool {
    process.mailbox.lock().borrow().len() == 0
}

pub fn has_message(process: &ProcessControlBlock, data: Term) -> bool {
    process.mailbox.lock().borrow().iter().any(|message| {
        &data
            == match message {
                Message::Process(message::Process { data }) => data,
                Message::HeapFragment(message::HeapFragment { data, .. }) => data,
            }
    })
}

pub fn has_heap_message(process: &ProcessControlBlock, data: Term) -> bool {
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

pub fn has_process_message(process: &ProcessControlBlock, data: Term) -> bool {
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

pub fn monitor_count(process: &ProcessControlBlock) -> usize {
    process.monitor_by_reference.lock().len()
}

pub fn monitored_count(process: &ProcessControlBlock) -> usize {
    process.monitored_pid_by_reference.lock().len()
}

pub fn receive_message(process: &ProcessControlBlock) -> Option<Term> {
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
