use std::sync::Arc;

use hashbrown::hash_map::HashMap;
use lazy_static::lazy_static;

use liblumen_core::locks::RwLock;

use liblumen_alloc::erts::exception::AllocResult;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::Code;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{Arity, ModuleFunctionArity};

pub fn contains_key(module: &Atom, function: &Atom, arity: Arity) -> bool {
    RW_LOCK_CODE_BY_ARITY_BY_FUNCTION_BY_MODULE
        .read()
        .get(module)
        .and_then(|code_by_arity_by_function| {
            code_by_arity_by_function
                .get(function)
                .map(|code_by_arity| code_by_arity.contains_key(&arity))
        })
        .unwrap_or(false)
}

pub fn display() -> String {
    let mut string = String::new();
    string.push_str("Exported code:\n");
    let code_by_arity_by_function_by_module = RW_LOCK_CODE_BY_ARITY_BY_FUNCTION_BY_MODULE.read();

    let mut modules: Vec<&Atom> = code_by_arity_by_function_by_module.keys().collect();
    modules.sort();

    for module in modules {
        string.push_str(&format!("  {}:\n", module));

        let code_by_arity_by_function = code_by_arity_by_function_by_module.get(module).unwrap();
        let mut functions: Vec<&Atom> = code_by_arity_by_function.keys().collect();
        functions.sort();

        for function in functions {
            let code_by_arity = code_by_arity_by_function.get(function).unwrap();
            let arities: Vec<&Arity> = code_by_arity.keys().collect();

            for arity in arities {
                string.push_str(&format!("    {}/{}\n", function, arity));
            }
        }
    }

    string
}

pub fn get(module: &Atom, function: &Atom, arity: Arity) -> Option<Code> {
    RW_LOCK_CODE_BY_ARITY_BY_FUNCTION_BY_MODULE
        .read()
        .get(module)
        .and_then(|code_by_arity_by_function| {
            code_by_arity_by_function
                .get(function)
                .and_then(|code_by_arity| code_by_arity.get(&arity).map(|code| *code))
        })
}

pub fn insert(module: Atom, function: Atom, arity: Arity, code: Code) {
    RW_LOCK_CODE_BY_ARITY_BY_FUNCTION_BY_MODULE
        .write()
        .entry(module)
        .or_insert_with(Default::default)
        .entry(function)
        .or_insert_with(Default::default)
        .insert(arity, code);
}

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    module: Atom,
    function: Atom,
    arity: Arity,
    code: Code,
    argument_vec: Vec<Term>,
) -> AllocResult<()> {
    assert_eq!(argument_vec.len(), arity as usize);
    for argument in argument_vec.iter().rev() {
        process.stack_push(*argument)?;
    }

    let module_function_arity = Arc::new(ModuleFunctionArity {
        module,
        function,
        arity,
    });
    let frame = Frame::new(module_function_arity, code);
    process.place_frame(frame, placement);

    Ok(())
}

lazy_static! {
    static ref RW_LOCK_CODE_BY_ARITY_BY_FUNCTION_BY_MODULE: RwLock<HashMap<Atom, HashMap<Atom, HashMap<u8, Code>>>> =
        Default::default();
}
