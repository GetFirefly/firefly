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

use lumen_runtime::atom::Existence::DoNotCare;
use lumen_runtime::heap::CloneIntoHeap;
use lumen_runtime::message::{self, Message};
use lumen_runtime::otp::erlang;
use lumen_runtime::process::stack::frame::Frame;
use lumen_runtime::process::{IntoProcess, ModuleFunctionArity, Process};
use lumen_runtime::term::Term;

use crate::elixir;

/// ```elixir
/// def counter(next_pid) do
///   receive do
///     n ->
///       send next_pid, n + 1
///   end
/// end
/// ```
pub fn counter_0_code(arc_process: &Arc<Process>) {
    // because there is a guardless match in the receive block, the first message will always be
    // removed and no loop is necessary
    let option_message = arc_process.mailbox.lock().unwrap().pop();

    arc_process.reduce();

    match option_message {
        Some(message) => {
            let n = match message {
                Message::Process(term) => term.clone(),
                Message::Heap(message::Heap { term, .. }) => {
                    let locked_heap = arc_process.heap.lock().unwrap();
                    term.clone_into_heap(&locked_heap)
                }
            };

            let counter_module_function_arity =
                arc_process.current_module_function_arity().unwrap();
            let frame_argument_vec = arc_process.pop_arguments(1);
            let next_pid = frame_argument_vec[0];

            let mut counter_2_frame =
                Frame::new(Arc::clone(&counter_module_function_arity), counter_2_code);
            counter_2_frame.push(next_pid);
            arc_process.replace_frame(counter_2_frame);

            let mut counter_1_frame = Frame::new(counter_module_function_arity, counter_1_code);
            counter_1_frame.push(n);
            arc_process.push_frame(counter_1_frame);

            Process::call_code(arc_process);
        }
        None => Arc::clone(arc_process).wait(),
    }
}

/// ```elixir
/// n + 1
/// ```
fn counter_1_code(arc_process: &Arc<Process>) {
    let frame_argument_vec = arc_process.pop_arguments(1);
    let n = frame_argument_vec[0];

    Process::tail_call_bif(
        arc_process,
        Term::str_to_atom("erlang", DoNotCare).unwrap(),
        Term::str_to_atom("+", DoNotCare).unwrap(),
        2,
        || erlang::add_2(n, 1.into_process(arc_process), arc_process),
    );
}

/// ```elixir
/// send next_pid, n + 1
/// ```
fn counter_2_code(arc_process: &Arc<Process>) {
    let frame_argument_vec = arc_process.pop_arguments(2);
    // n + 1 is on the top of stack even though it is the second argument because it is the return
    // from `counter_1_code`
    let sum = frame_argument_vec[0];
    let next_pid = frame_argument_vec[1];

    Process::tail_call_bif(
        arc_process,
        Term::str_to_atom("erlang", DoNotCare).unwrap(),
        Term::str_to_atom("send", DoNotCare).unwrap(),
        2,
        || erlang::send_2(next_pid, sum, arc_process),
    );
}

fn create_processes_frame_with_arguments(n: Term) -> Frame {
    let module_function_arity = Arc::new(ModuleFunctionArity {
        module: Term::str_to_atom("Elixir.Chain", DoNotCare).unwrap(),
        function: Term::str_to_atom("create_processes", DoNotCare).unwrap(),
        arity: 1,
    });

    let mut frame = Frame::new(module_function_arity, create_processes_0_code);
    frame.push(n);

    frame
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
fn create_processes_0_code(arc_process: &Arc<Process>) {
    let frame_argument_vec = arc_process.pop_arguments(1);
    let n = frame_argument_vec[0];

    // assumed to be fast enough to act as a BIF
    let result = elixir::range::new(1.into_process(arc_process), n, arc_process);

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

            let reducer = create_processes_reducer_function(arc_process);

            let enum_reduce_frame =
                elixir::r#enum::reduce_frame_with_arguments(range, arc_process.pid, reducer);
            arc_process.push_frame(enum_reduce_frame);

            Process::call_code(arc_process);
        }
        Err(exception) => arc_process.exception(exception),
    }
}

fn create_processes_reducer_function(process: &Process) -> Term {
    let module_function_arity = Arc::new(ModuleFunctionArity {
        module: Term::str_to_atom("Elixir.Chain", DoNotCare).unwrap(),
        function: Term::str_to_atom("create_processes_reducer", DoNotCare).unwrap(),
        arity: 2,
    });

    Term::function(
        module_function_arity,
        create_processes_reducer_0_code,
        process,
    )
}

fn create_processes_reducer_0_code(arc_process: &Arc<Process>) {
    let frame_argument_vec = arc_process.pop_arguments(2);
    let _element = frame_argument_vec[0];
    let send_to = frame_argument_vec[1];

    let module = Term::str_to_atom("Elixir.Chain", DoNotCare).unwrap();
    let function = Term::str_to_atom("counter", DoNotCare).unwrap();
    let arguments = Term::slice_to_list(&[send_to], arc_process);

    // In `lumen` compiled code the compile would optimize this to a direct call of
    // `Scheduler::spawn(arc_process, module, function, arguments, counter_0_code)`, but we want
    // to demonstrate the the `lumen_runtime::code::set_apply_fn` system works here.
    match erlang::spawn_3(module, function, arguments, arc_process) {
        Ok(child_pid) => {
            arc_process.reduce();
            arc_process.return_from_call(child_pid);
            Process::call_code(arc_process);
        }
        Err(exception) => {
            arc_process.reduce();
            arc_process.exception(exception);
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
fn create_processes_1_code(arc_process: &Arc<Process>) {
    let frame_argument_vec = arc_process.pop_arguments(1);

    // placed on top of stack by return from `elixir::r#enum::reduce_0_code`
    let last = frame_argument_vec[0];
    #[cfg(debug_assertions)]
    debug_assert!(last.is_local_pid(), "last ({:?}) is not a local pid", last);

    match erlang::send_2(last, 0.into_process(arc_process), arc_process) {
        Ok(_) => {
            arc_process.reduce();

            let module_function_arity = arc_process.current_module_function_arity().unwrap();
            let create_processes_2_frame =
                Frame::new(module_function_arity, create_processes_2_code);
            arc_process.replace_frame(create_processes_2_frame);

            Process::call_code(arc_process);
        }
        Err(exception) => {
            arc_process.reduce();
            arc_process.exception(exception);
        }
    }
}

/// ```elixir
/// receive do # and wait for the result to come back to us
///   final_answer when is_integer(final_answer) ->
///     "Result is #{inspect(final_answer)}"
/// end
/// ```
fn create_processes_2_code(arc_process: &Arc<Process>) {
    // locked mailbox scope
    let received = {
        let mut locked_mailbox = arc_process.mailbox.lock().unwrap();
        let seen = locked_mailbox.seen();
        let mut found_position = None;

        for (position, message) in locked_mailbox.iter().enumerate() {
            if seen < (position as isize) {
                let message_term = match message {
                    Message::Process(term) => term,
                    Message::Heap(message::Heap { term, .. }) => term,
                };

                if message_term.is_integer() {
                    let process_term = match message {
                        Message::Process(message_term) => message_term.clone(),
                        Message::Heap { .. } => {
                            let locked_heap = arc_process.heap.lock().unwrap();
                            message_term.clone_into_heap(&locked_heap)
                        }
                    };

                    let module_function_arity =
                        arc_process.current_module_function_arity().unwrap();
                    let mut frame = Frame::new(module_function_arity, create_processes_3_code);
                    frame.push(process_term);
                    arc_process.replace_frame(frame);

                    found_position = Some(position);

                    break;
                }
            }
        }

        // separate because can't remove during iteration
        match found_position {
            Some(position) => {
                locked_mailbox.remove(position);
                locked_mailbox.unmark_seen();

                true
            }
            None => {
                locked_mailbox.mark_seen();

                false
            }
        }
    };

    arc_process.reduce();

    if received {
        Process::call_code(arc_process);
    } else {
        arc_process.wait()
    }
}

/// ```elixir
/// "Result is #{inspect(final_answer)}"
/// ```
fn create_processes_3_code(arc_process: &Arc<Process>) {
    let frame_argument_vec = arc_process.pop_arguments(1);
    // set by the receive in `create_processes_2_code`
    let final_answer = frame_argument_vec[0];

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

    let elixir_string = Term::slice_to_binary(formatted.as_bytes(), arc_process);

    arc_process.return_from_call(elixir_string);
    Process::call_code(arc_process);
}

/// ```elixir
/// def run(n) do
///   IO.puts(inspect(:timer.tc(Chain, :create_processes, [n])))
/// end
pub fn run_0_code(arc_process: &Arc<Process>) {
    web_sys::console::time_with_label("Chain.run");
    arc_process.reduce();

    let frame_argument_vec = arc_process.pop_arguments(1);
    let n = frame_argument_vec[0];

    let run_module_function_arity = arc_process.current_module_function_arity().unwrap();

    // run_1_code will put `elixir_string` on top of stack, which matches the order needed by
    // elixir::io::pust_code, so it can be called directly instead of through a `run_N_code`
    let io_puts_frame = elixir::io::puts_frame();
    arc_process.replace_frame(io_puts_frame);

    let run_code_1_frame = Frame::new(run_module_function_arity, run_1_code);
    arc_process.push_frame(run_code_1_frame);

    let create_processes_frame = create_processes_frame_with_arguments(n);
    arc_process.push_frame(create_processes_frame);

    Process::call_code(arc_process);
}

fn run_1_code(arc_process: &Arc<Process>) {
    let frame_argument_vec = arc_process.pop_arguments(1);
    let elixir_string = frame_argument_vec[0];

    web_sys::console::time_end_with_label("Chain.run");
    arc_process.reduce();

    arc_process.return_from_call(elixir_string);
    Process::call_code(arc_process);
}
