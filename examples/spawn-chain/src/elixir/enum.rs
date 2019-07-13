use std::sync::Arc;

use lumen_runtime::code::Code;
use lumen_runtime::otp::erlang;

pub fn reduce_frame_with_arguments(enumerable: Term, initial: Term, reducer: Term) -> Frame {
    let module_function_arity = Arc::new(ModuleFunctionArity {
        module: Term::str_to_atom("Elixir.Enum", DoNotCare).unwrap(),
        function: Term::str_to_atom("reduce", DoNotCare).unwrap(),
        arity: 3,
    });
    let mut frame = Frame::new(module_function_arity, reduce_0_code);
    frame.push(reducer);
    frame.push(initial);
    frame.push(enumerable);

    frame
}

fn reduce_0_code(arc_process: &Arc<Process>) {
    let frame_argument_vec = arc_process.pop_arguments(3);
    let enumerable = frame_argument_vec[0];
    let initial = frame_argument_vec[1];
    let reducer = frame_argument_vec[2];

    match enumerable.tag() {
        Boxed => {
            let unboxed_enumerable: &Term = enumerable.unbox_reference();

            match unboxed_enumerable.tag() {
                Map => {
                    let map: &Map = enumerable.unbox_reference();

                    match map.get(Term::str_to_atom("__struct__", DoNotCare).unwrap()) {
                        Some(struct_name) => {
                            if struct_name == Term::str_to_atom("Elixir.Range", DoNotCare).unwrap()
                            {
                                // This assumes no one was cheeky and messed with the map
                                // representation of the struct
                                let first_key = Term::str_to_atom("first", DoNotCare).unwrap();
                                let first = map.get(first_key).unwrap();

                                let last_key = Term::str_to_atom("last", DoNotCare).unwrap();
                                let last = map.get(last_key).unwrap();

                                arc_process.reduce();

                                let reduce_range_frame = reduce_range_frame_with_arguments(
                                    first, last, initial, reducer,
                                );
                                arc_process.replace_frame(reduce_range_frame);

                                Process::call_code(arc_process);
                            } else {
                                arc_process.reduce();
                                arc_process.exception(lumen_runtime::badarg!())
                            }
                        }
                        None => {
                            arc_process.reduce();
                            arc_process.exception(lumen_runtime::badarg!())
                        }
                    }
                }
                _ => unimplemented!(),
            }
        }
        _ => {
            arc_process.reduce();
            arc_process.exception(lumen_runtime::badarg!())
        }
    }
}

fn reduce_range_frame_with_arguments(
    first: Term,
    last: Term,
    initial: Term,
    reducer: Term,
) -> Frame {
    let (function_name, code): (&str, Code) = if first <= last {
        ("reduce_range_inc", reduce_range_inc_0_code)
    } else {
        ("reduce_range_dec", reduce_range_dec_0_code)
    };

    let function = Term::str_to_atom(function_name, DoNotCare).unwrap();
    let module_function_arity = Arc::new(ModuleFunctionArity {
        module: Term::str_to_atom("Elixir.Enum", DoNotCare).unwrap(),
        function,
        arity: 4,
    });

    let mut frame = Frame::new(module_function_arity, code);
    frame.push(reducer);
    frame.push(initial);
    frame.push(last);
    frame.push(first);

    frame
}

/// ```elixir
/// defp reduce_range_inc(first, first, acc, fun) do
///   fun.(first, acc)
/// end
///
/// defp reduce_range_inc(first, last, acc, fun) do
///   reduce_range_inc(first + 1, last, fun.(first, acc), fun)
/// end
/// ```
fn reduce_range_inc_0_code(arc_process: &Arc<Process>) {
    let frame_argument_vec = arc_process.pop_arguments(4);
    let first = frame_argument_vec[0];
    let last = frame_argument_vec[1];
    let acc = frame_argument_vec[2];
    let reducer = frame_argument_vec[3];

    arc_process.reduce();

    // defp reduce_range_inc(first, first, acc, fun) do
    //   fun.(first, acc)
    // end
    if first == last {
        match reducer.tag() {
            Boxed => {
                let unboxed_reducer: &Term = reducer.unbox_reference();

                match unboxed_reducer.tag() {
                    Function => {
                        let function: &Function = reducer.unbox_reference();

                        match function.frame_with_arguments(vec![first, acc]) {
                            Some(function_frame) => {
                                arc_process.replace_frame(function_frame);

                                Process::call_code(arc_process);
                            }
                            None => arc_process.exception(lumen_runtime::badarity!(
                                reducer,
                                Term::slice_to_list(&[first, acc], arc_process),
                                arc_process
                            )),
                        }
                    }
                    _ => arc_process.exception(lumen_runtime::badfun!(reducer, arc_process)),
                }
            }
            _ => arc_process.exception(lumen_runtime::badfun!(reducer, arc_process)),
        }
    }
    // defp reduce_range_inc(first, last, acc, fun) do
    //   reduce_range_inc(first + 1, last, fun.(first, acc), fun)
    // end
    else {
        let mut reduce_range_inc_1_frame = Frame::new(
            arc_process.current_module_function_arity().unwrap(),
            reduce_range_inc_1_code,
        );
        reduce_range_inc_1_frame.push(reducer);
        reduce_range_inc_1_frame.push(acc);
        reduce_range_inc_1_frame.push(last);
        reduce_range_inc_1_frame.push(first);
        arc_process.replace_frame(reduce_range_inc_1_frame);

        Process::call_code(arc_process);
    }
}

/// defp reduce_range_inc(first, last, acc, fun) do
///   reduce_range_inc(first + 1, last, fun.(first, acc), fun)
/// end
fn reduce_range_inc_1_code(arc_process: &Arc<Process>) {
    let frame_argument_vec = arc_process.pop_arguments(4);
    let first = frame_argument_vec[0];
    let last = frame_argument_vec[1];
    let acc = frame_argument_vec[2];
    let reducer = frame_argument_vec[3];

    arc_process.reduce();

    match erlang::add_2(first, 1.into_process(arc_process), arc_process) {
        Ok(sum) => match reducer.tag() {
            Boxed => {
                let unboxed_reducer: &Term = reducer.unbox_reference();

                match unboxed_reducer.tag() {
                    Function => {
                        let mut reduce_range_inc_2_frame = Frame::new(
                            arc_process.current_module_function_arity().unwrap(),
                            reduce_range_inc_2_code,
                        );
                        reduce_range_inc_2_frame.push(reducer);
                        reduce_range_inc_2_frame.push(last);
                        reduce_range_inc_2_frame.push(sum);
                        arc_process.replace_frame(reduce_range_inc_2_frame);

                        let function: &Function = reducer.unbox_reference();

                        match function.frame_with_arguments(vec![first, acc]) {
                            Some(function_frame) => {
                                arc_process.push_frame(function_frame);

                                Process::call_code(arc_process);
                            }
                            None => arc_process.exception(lumen_runtime::badarity!(
                                reducer,
                                Term::slice_to_list(&[first, acc], arc_process),
                                arc_process
                            )),
                        }
                    }
                    _ => arc_process.exception(lumen_runtime::badfun!(reducer, arc_process)),
                }
            }
            _ => arc_process.exception(lumen_runtime::badfun!(reducer, arc_process)),
        },
        Err(exception) => {
            arc_process.exception(exception);
        }
    }
}

fn reduce_range_inc_2_code(arc_process: &Arc<Process>) {
    let frame_argument_vec = arc_process.pop_arguments(4);
    // acc is on top of stack because it is the return from `reducer` call
    let acc = frame_argument_vec[0];
    let first = frame_argument_vec[1];
    let last = frame_argument_vec[2];
    let reducer = frame_argument_vec[3];

    arc_process.reduce();

    let mut reduce_range_inc_0_frame = Frame::new(
        arc_process.current_module_function_arity().unwrap(),
        reduce_range_inc_0_code,
    );
    reduce_range_inc_0_frame.push(reducer);
    reduce_range_inc_0_frame.push(acc);
    reduce_range_inc_0_frame.push(last);
    reduce_range_inc_0_frame.push(first);
    arc_process.replace_frame(reduce_range_inc_0_frame);

    Process::call_code(arc_process);
}

fn reduce_range_dec_0_code(_arc_process: &Arc<Process>) {
    unimplemented!()
}
