//! ```elixir
//! defmodule Chain do
//!   def counter(next_pid) do
//!     receive do
//!       n ->
//!         send next_pid, n + 1
//!     end
//!   end
//!
//!   def create_processes(n) do
//!     last =
//!       Enum.reduce(
//!         1..n,
//!         self,
//!         fn (_, send_to) ->
//!           spawn(Chain, :counter, [send_to])
//!         end
//!       )
//!
//!     send(last, 0) # start the count by sending a zero to the last process
//!
//!     receive do # and wait for the result to come back to us
//!       final_answer when is_integer(final_answer) ->
//!         "Result is #{inspect(final_answer)}"
//!     end
//!   end
//!
//!   def run(n) do
//!     IO.puts(inspect(:timer.tc(Chain, :create_processes, [n])))
//!   end
//! end
//! ```

use std::convert::TryInto;
use std::sync::Arc;

use num_bigint::BigInt;

use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::exception::Exception;
use liblumen_alloc::erts::process::code::stack::frame::Frame;
use liblumen_alloc::erts::process::code::{result_from_exception, Result};
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::term::{atom_unchecked, Atom, Term};
use liblumen_alloc::erts::ModuleFunctionArity;
use liblumen_alloc::CloneToProcess;

use lumen_runtime::code::tail_call_bif;
use lumen_runtime::otp::erlang;
use lumen_runtime::system;

use crate::elixir;

/// ```elixir
/// def counter(next_pid) do
///   receive do
///     n ->
///       send next_pid, n + 1
///   end
/// end
/// ```
pub fn counter_0_code(arc_process: &Arc<ProcessControlBlock>) -> Result {
    arc_process.reduce();

    // Because there is a guardless match in the receive block, the first message will always be
    // removed and no loop is necessary.
    //
    // CANNOT be in `match` as it will hold temporaries in `match` arms causing a `park`.
    let received = arc_process
        .mailbox
        .lock()
        .borrow_mut()
        .receive(&mut arc_process.acquire_heap());

    match received {
        Some(Ok(n)) => {
            let counter_module_function_arity =
                arc_process.current_module_function_arity().unwrap();
            let next_pid = arc_process.stack_pop().unwrap();

            let counter_2_frame =
                Frame::new(Arc::clone(&counter_module_function_arity), counter_2_code);
            arc_process.stack_push(next_pid)?;
            arc_process.replace_frame(counter_2_frame);

            let counter_1_frame = Frame::new(counter_module_function_arity, counter_1_code);
            arc_process.stack_push(n)?;
            arc_process.push_frame(counter_1_frame);

            ProcessControlBlock::call_code(arc_process)
        }
        None => Ok(Arc::clone(arc_process).wait()),
        Some(Err(alloc_err)) => Err(alloc_err.into()),
    }
}

/// ```elixir
/// n + 1
/// ```
fn counter_1_code(arc_process: &Arc<ProcessControlBlock>) -> Result {
    let n = arc_process.stack_pop().unwrap();
    assert!(
        n.is_integer(),
        "{:?}: {:?} is not an integer",
        arc_process,
        n
    );

    tail_call_bif(
        arc_process,
        Atom::try_from_str("erlang").unwrap(),
        Atom::try_from_str("+").unwrap(),
        2,
        || erlang::add_2(n, arc_process.integer(1)?, arc_process),
    )
}

/// ```elixir
/// send next_pid, n + 1
/// ```
fn counter_2_code(arc_process: &Arc<ProcessControlBlock>) -> Result {
    // n + 1 is on the top of stack even though it is the second argument because it is the return
    // from `counter_1_code`
    let sum = arc_process.stack_pop().unwrap();
    let next_pid = arc_process.stack_pop().unwrap();

    tail_call_bif(
        arc_process,
        Atom::try_from_str("erlang").unwrap(),
        Atom::try_from_str("send").unwrap(),
        2,
        || {
            // printing is slow, so print to show progress, but not too often that is slows down a
            // lot.
            if arc_process.pid().number() % 1_000 == 0 {
                system::io::puts(&format!(
                    "{:?} Sending {:?} to {:?}",
                    arc_process, sum, next_pid
                ));
            }

            match erlang::send_2(next_pid, sum, arc_process) {
                Ok(term) => Ok(term),
                Err(error) => {
                    system::io::puts(&format!(
                        "[{}:{}] {:?} send({:?}, {:?})\n{:?}",
                        file!(),
                        line!(),
                        arc_process,
                        next_pid,
                        sum,
                        *arc_process.acquire_heap()
                    ));
                    Err(error)
                }
            }
        },
    )
}

/// ```elixir
/// def create_processes(n) do
///   last =
///     Enum.reduce(
///       1..n,
///       self,
///       fn (_, send_to) ->
///         spawn(Chain, :counter, [send_to])
///       end
///     )
///
///   send(last, 0) # start the count by sending a zero to the last process
///
///   receive do # and wait for the result to come back to us
///     final_answer when is_integer(final_answer) ->
///       "Result is #{inspect(final_answer)}"
///   end
/// end
/// ```
fn create_processes_0_code(arc_process: &Arc<ProcessControlBlock>) -> Result {
    let n = arc_process.stack_pop().unwrap();

    // assumed to be fast enough to act as a BIF
    let result = elixir::range::new(arc_process.integer(1)?, n, arc_process);

    arc_process.reduce();

    match result {
        Ok(range) => {
            let create_processes_module_function_arity =
                arc_process.current_module_function_arity().unwrap();
            let create_processes_1_code_frame = Frame::new(
                create_processes_module_function_arity,
                create_processes_1_code,
            );
            arc_process.replace_frame(create_processes_1_code_frame);

            let reducer = create_processes_reducer_function(arc_process)?;

            let module_function_arity = Arc::new(ModuleFunctionArity {
                module: Atom::try_from_str("Elixir.Enum").unwrap(),
                function: Atom::try_from_str("reduce").unwrap(),
                arity: 3,
            });
            let enum_reduce_frame =
                Frame::new(module_function_arity, elixir::r#enum::reduce_0_code);
            arc_process.stack_push(reducer)?;
            arc_process.stack_push(arc_process.pid_term())?;
            arc_process.stack_push(range)?;

            arc_process.push_frame(enum_reduce_frame);

            ProcessControlBlock::call_code(arc_process)
        }
        Err(exception) => result_from_exception(arc_process, exception),
    }
}

fn create_processes_reducer_function(
    process: &ProcessControlBlock,
) -> std::result::Result<Term, Alloc> {
    let module_function_arity = Arc::new(ModuleFunctionArity {
        module: Atom::try_from_str("Elixir.Chain").unwrap(),
        function: Atom::try_from_str("create_processes_reducer").unwrap(),
        arity: 2,
    });

    process.closure(
        process.pid_term(),
        module_function_arity,
        create_processes_reducer_0_code,
    )
}

fn create_processes_reducer_0_code(arc_process: &Arc<ProcessControlBlock>) -> Result {
    let _element = arc_process.stack_pop().unwrap();
    let send_to = arc_process.stack_pop().unwrap();

    let module = atom_unchecked("Elixir.Chain");
    let function = atom_unchecked("counter");
    let arguments = arc_process.list_from_slice(&[send_to])?;

    // In `lumen` compiled code the compile would optimize this to a direct call of
    // `Scheduler::spawn(arc_process, module, function, arguments, counter_0_code)`, but we want
    // to demonstrate the the `lumen_runtime::code::set_apply_fn` system works here.
    match erlang::spawn_3(module, function, arguments, arc_process) {
        Ok(child_pid) => {
            arc_process.reduce();
            arc_process.return_from_call(child_pid)?;
            ProcessControlBlock::call_code(arc_process)
        }
        Err(exception) => {
            arc_process.reduce();

            match exception {
                Exception::Runtime(runtime_exception) => {
                    arc_process.exception(runtime_exception);

                    Ok(())
                }
                Exception::System(system_exception) => Err(system_exception),
            }
        }
    }
}

/// ```elixir
/// send(last, 0) # start the count by sending a zero to the last process
///
/// receive do # and wait for the result to come back to us
///   final_answer when is_integer(final_answer) ->
///     "Result is #{inspect(final_answer)}"
/// end
/// ```
fn create_processes_1_code(arc_process: &Arc<ProcessControlBlock>) -> Result {
    // placed on top of stack by return from `elixir::r#enum::reduce_0_code`
    let last = arc_process.stack_pop().unwrap();
    assert!(last.is_local_pid(), "last ({:?}) is not a local pid", last);

    match erlang::send_2(last, arc_process.integer(0)?, arc_process) {
        Ok(_) => {
            arc_process.reduce();

            let module_function_arity = arc_process.current_module_function_arity().unwrap();
            let create_processes_2_frame =
                Frame::new(module_function_arity, create_processes_2_code);
            arc_process.replace_frame(create_processes_2_frame);

            ProcessControlBlock::call_code(arc_process)
        }
        Err(exception) => {
            arc_process.reduce();

            match exception {
                Exception::Runtime(runtime_exception) => {
                    arc_process.exception(runtime_exception);

                    Ok(())
                }
                Exception::System(system_exception) => Err(system_exception),
            }
        }
    }
}

/// ```elixir
/// receive do # and wait for the result to come back to us
///   final_answer when is_integer(final_answer) ->
///     "Result is #{inspect(final_answer)}"
/// end
/// ```
fn create_processes_2_code(arc_process: &Arc<ProcessControlBlock>) -> Result {
    // locked mailbox scope
    let received = {
        let mailbox_guard = arc_process.mailbox.lock();
        let mut mailbox = mailbox_guard.borrow_mut();
        let seen = mailbox.seen();
        let mut found_position = None;

        for (position, message) in mailbox.iter().enumerate() {
            if seen < (position as isize) {
                let message_data = message.data();

                if message_data.is_integer() {
                    let process_term = if message.is_on_heap() {
                        message.data()
                    } else {
                        match message
                            .data()
                            .clone_to_heap(&mut arc_process.acquire_heap())
                        {
                            Ok(heap_data) => heap_data,
                            Err(alloc_err) => {
                                arc_process.reduce();

                                return Err(alloc_err.into());
                            }
                        }
                    };

                    let module_function_arity =
                        arc_process.current_module_function_arity().unwrap();
                    let frame = Frame::new(module_function_arity, create_processes_3_code);
                    arc_process.stack_push(process_term)?;
                    arc_process.replace_frame(frame);

                    found_position = Some(position);

                    break;
                } else {
                    // NOT in original Elixir source and would not be in compiled code, but helps
                    // debug runtime bugs leading to deadlocks.
                    panic!(
                        "Non-integer message ({:?}) received in {:?}",
                        message_data, arc_process
                    );
                }
            }
        }

        // separate because can't remove during iteration
        match found_position {
            Some(position) => {
                mailbox.remove(position);
                mailbox.unmark_seen();

                true
            }
            None => {
                mailbox.mark_seen();

                false
            }
        }
    };

    arc_process.reduce();

    if received {
        ProcessControlBlock::call_code(arc_process)
    } else {
        arc_process.wait();

        Ok(())
    }
}

/// ```elixir
/// "Result is #{inspect(final_answer)}"
/// ```
fn create_processes_3_code(arc_process: &Arc<ProcessControlBlock>) -> Result {
    // set by the receive in `create_processes_2_code`
    let final_answer = arc_process.stack_pop().unwrap();

    // TODO this would be replaced by what interpolation really does in Elixir in
    // `lumen` compiled code.
    #[allow(unused_variables)]
    let big_int: BigInt = match final_answer.try_into() {
        Ok(big_int) => big_int,
        Err(exception) => {
            #[cfg(debug_assertions)]
            panic!("{:?}", exception);
            #[cfg(not(debug_assertions))]
            panic!("Final answer cannot be converted to a BigInt");
        }
    };
    let formatted = format!("Result is {:}", big_int);

    arc_process.reduce();

    let elixir_string = arc_process.binary_from_bytes(formatted.as_bytes())?;

    arc_process.return_from_call(elixir_string)?;
    ProcessControlBlock::call_code(arc_process)
}

/// ```elixir
/// def run(n) do
///   IO.puts(inspect(:timer.tc(Chain, :create_processes, [n])))
/// end
pub fn run_0_code(arc_process: &Arc<ProcessControlBlock>) -> Result {
    #[cfg(target_arch = "wasm32")]
    web_sys::console::time_with_label("Chain.run");
    arc_process.reduce();

    let n = arc_process.stack_pop().unwrap();

    let run_module_function_arity = arc_process.current_module_function_arity().unwrap();

    // run_1_code will put `elixir_string` on top of stack, which matches the order needed by
    // elixir::io::pust_code, so it can be called directly instead of through a `run_N_code`
    let io_puts_frame = elixir::io::puts_frame();
    arc_process.replace_frame(io_puts_frame);

    let run_code_1_frame = Frame::new(run_module_function_arity, run_1_code);
    arc_process.push_frame(run_code_1_frame);

    let module_function_arity = Arc::new(ModuleFunctionArity {
        module: Atom::try_from_str("Elixir.Chain").unwrap(),
        function: Atom::try_from_str("create_processes").unwrap(),
        arity: 1,
    });

    let create_processes_frame = Frame::new(module_function_arity, create_processes_0_code);
    arc_process.stack_push(n)?;
    arc_process.push_frame(create_processes_frame);

    ProcessControlBlock::call_code(arc_process)
}

fn run_1_code(arc_process: &Arc<ProcessControlBlock>) -> Result {
    let elixir_string = arc_process.stack_pop().unwrap();

    #[cfg(target_arch = "wasm32")]
    web_sys::console::time_end_with_label("Chain.run");
    arc_process.reduce();

    arc_process.return_from_call(elixir_string)?;
    ProcessControlBlock::call_code(arc_process)
}
