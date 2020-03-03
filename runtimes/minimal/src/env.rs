use core::slice;
use std::env::ArgsOs;

use once_cell::sync::OnceCell;

use liblumen_alloc::erts::process::{Process, ProcessFlags};
use liblumen_alloc::erts::term::prelude::*;

use crate::scheduler::Scheduler;

static ARGV: OnceCell<Vec<String>> = OnceCell::new();
static ARGV_TERM: OnceCell<Vec<BinaryLiteral>> = OnceCell::new();

pub(crate) fn init_argv_from_slice(argv: ArgsOs) -> anyhow::Result<()> {
    use liblumen_alloc::erts::string::Encoding;

    let mut args = Vec::with_capacity(argv.len());
    for arg in argv {
        args.push(arg.to_string_lossy().into_owned());
    }
    ARGV.set(args).unwrap();

    let argv = ARGV.get().map(|v| v.as_slice()).unwrap();
    let mut literals = Vec::with_capacity(argv.len());
    for arg in argv {
        let bytes = arg.as_bytes();
        literals.push(BinaryLiteral::from_raw_bytes(
            bytes.as_ptr() as *mut u8,
            bytes.len(),
            Some(Encoding::Utf8),
        ));
    }

    ARGV_TERM.set(literals).unwrap();

    Ok(())
}

#[allow(dead_code)]
pub(crate) fn init_argv(argv: *const *const libc::c_char, argc: u32) -> anyhow::Result<()> {
    use liblumen_alloc::erts::string::Encoding;
    use std::ffi::CStr;

    let argc = argc as usize;
    if argc == 0 {
        ARGV.set(Vec::new()).unwrap();
        return Ok(());
    }

    let mut args = Vec::with_capacity(argc);
    let mut literals = Vec::with_capacity(argc);
    unsafe {
        let argv_slice = slice::from_raw_parts::<'static>(argv, argc as usize);

        for arg_ptr in argv_slice.iter().copied() {
            let cs = CStr::from_ptr::<'static>(arg_ptr);
            match cs.to_str() {
                Ok(s) => {
                    let bytes = cs.to_bytes();
                    args.push(s.to_string());
                    literals.push(BinaryLiteral::from_raw_bytes(
                        bytes.as_ptr() as *mut u8,
                        bytes.len(),
                        Some(Encoding::Utf8),
                    ));
                }
                Err(_) => {
                    let bytes = cs.to_bytes();
                    let encoding = if bytes.is_ascii() {
                        Some(Encoding::Latin1)
                    } else {
                        None
                    };
                    args.push(cs.to_string_lossy().into_owned());
                    literals.push(BinaryLiteral::from_raw_bytes(
                        bytes.as_ptr() as *mut u8,
                        bytes.len(),
                        encoding,
                    ));
                }
            }
        }
    }

    ARGV.set(args).unwrap();
    ARGV_TERM.set(literals).unwrap();

    Ok(())
}

pub fn get_argv<'a>() -> Option<&'a [String]> {
    ARGV.get().map(|v| v.as_slice())
}

pub fn get_argv_literals<'a>() -> Option<&'a [BinaryLiteral]> {
    ARGV_TERM.get().map(|v| v.as_slice())
}

#[export_name = "init:get_plain_arguments/0"]
pub extern "C" fn get_plain_arguments() -> Term {
    get_plain_arguments_with_process(&Scheduler::current_process())
}

fn get_plain_arguments_with_process(process: &Process) -> Term {
    let argv = get_argv_literals();
    if argv.is_none() {
        return Term::NIL;
    }
    let argv = argv.unwrap();
    if argv.len() == 0 {
        return Term::NIL;
    }

    let mut heap = process.acquire_heap();
    let mut builder = ListBuilder::new(&mut heap);
    for arg in argv {
        let boxed: Boxed<BinaryLiteral> =
            unsafe { Boxed::new_unchecked(arg as *const _ as *mut _) };
        builder = builder.push(boxed.into());
    }

    if let Ok(cons) = builder.finish() {
        cons.into()
    } else {
        process.set_flags(ProcessFlags::GrowHeap | ProcessFlags::ForceGC);
        Term::NONE
    }
}
