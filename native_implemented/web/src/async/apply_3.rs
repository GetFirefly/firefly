use std::sync::Arc;

use js_sys::Promise;

use liblumen_core::locks::Mutex;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::fragment::HeapFragment;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::executor::{self, Executor};
use crate::promise;
use crate::runtime::process::spawn::options::Options;
use crate::runtime::scheduler;

use liblumen_alloc::erts::term::list::optional_cons_to_term;
use liblumen_otp::erlang::apply::arguments_term_to_vec;
use liblumen_otp::term_try_into_atom;

#[native_implemented::function(Elixir.Lumen.Web.Async:apply/3)]
fn result(
    process: &Process,
    module: Term,
    function: Term,
    arguments: Term,
) -> exception::Result<Term> {
    let module_atom = term_try_into_atom!(module)?;
    let function_atom = term_try_into_atom!(function)?;
    let argument_vec = arguments_term_to_vec(arguments)?;

    promise(module_atom, function_atom, argument_vec, Default::default())
        .map(|promise| promise::to_term(promise, process))
}

pub fn promise(
    module: Atom,
    function: Atom,
    argument_vec: Vec<Term>,
    options: Options,
) -> exception::Result<Promise> {
    // Lumen.Web.Async functions are assumed to be parent less to support spawning directly from
    // embedding Rust code
    let parent = None;

    let spawn_module = executor::module();
    let spawn_function = executor::apply_4::function();

    let mut executor = Executor::new();
    let promise = executor.promise();
    let (executor_boxed_resource, executor_non_null_heap_fragment) =
        HeapFragment::new_resource(Arc::new(Mutex::new(executor))).unwrap();
    let executor_resource = executor_boxed_resource.encode().unwrap();

    let (arguments, arguments_option_non_null_heap_fragment) = if argument_vec.len() == 0 {
        (Term::NIL, None)
    } else {
        let (arguments_option_boxed_cons, arguments_non_null_heap_fragment) =
            HeapFragment::new_list_from_slice(&argument_vec).unwrap();

        (
            optional_cons_to_term(arguments_option_boxed_cons),
            Some(arguments_non_null_heap_fragment),
        )
    };

    let spawn_arguments = vec![
        executor_resource,
        module.encode().unwrap(),
        function.encode().unwrap(),
        arguments,
    ];

    scheduler::current().spawn_module_function_arguments(
        parent,
        spawn_module,
        spawn_function,
        spawn_arguments,
        options,
    )?;

    // HeapFragments can be dropped after `spawn_module_function_arguments` has returned because all
    // arguments are cloned recursively to the spawned process.

    if let Some(arguments_non_null_heap_fragment) = arguments_option_non_null_heap_fragment {
        std::mem::drop(arguments_non_null_heap_fragment);
    }

    std::mem::drop(executor_non_null_heap_fragment);

    Ok(promise)
}
