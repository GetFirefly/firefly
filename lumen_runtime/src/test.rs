pub mod r#loop;

// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
pub mod strategy;

use liblumen_alloc::erts::message::{self, Message};
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::term::Term;

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
