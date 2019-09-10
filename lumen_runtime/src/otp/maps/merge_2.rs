// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;
use std::sync::Arc;

use hashbrown::HashMap;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{Atom, Boxed, Map, Term};
use liblumen_alloc::{badmap, ModuleFunctionArity};

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    map1: Term,
    map2: Term,
) -> Result<(), Alloc> {
    process.stack_push(map2)?;
    process.stack_push(map1)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Crate Public

pub(in crate::otp) fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let map1 = arc_process.stack_pop().unwrap();
    let map2 = arc_process.stack_pop().unwrap();

    match native(arc_process, map1, map2) {
        Ok(map3) => {
            arc_process.return_from_call(map3)?;

            Process::call_code(arc_process)
        }
        Err(exception) => result_from_exception(arc_process, exception),
    }
}

// Private

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("merge").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 2,
    })
}

pub fn native(process: &Process, map1: Term, map2: Term) -> exception::Result {
    let result_map1: Result<Boxed<Map>, _> = map1.try_into();

    match result_map1 {
        Ok(map1) => {
            let result_map2: Result<Boxed<Map>, _> = map2.try_into();

            match result_map2 {
                Ok(map2) => {
                    let hash_map1: &HashMap<_, _> = map1.as_ref();
                    let hash_map2: &HashMap<_, _> = map2.as_ref();

                    let mut hash_map3: HashMap<Term, Term> =
                        HashMap::with_capacity(hash_map1.len() + hash_map2.len());

                    for (key, value) in hash_map1 {
                        hash_map3.insert(*key, *value);
                    }

                    for (key, value) in hash_map2 {
                        hash_map3.insert(*key, *value);
                    }

                    process
                        .map_from_hash_map(hash_map3)
                        .map_err(|error| error.into())
                }
                Err(_) => Err(badmap!(process, map2)),
            }
        }
        Err(_) => Err(badmap!(process, map1)),
    }
}
