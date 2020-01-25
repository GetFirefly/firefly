pub mod init;

use hashbrown::hash_map::HashMap;

use liblumen_core::locks::RwLock;

use liblumen_alloc::erts::exception::AllocResult;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::LocatedCode;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::closure::Definition;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::Arity;

pub fn contains_key(module: &Atom, definition: &Definition, arity: Arity) -> bool {
    RW_LOCK_LOCATED_CODE_BY_ARITY_BY_DEFINITION_BY_MODULE
        .read()
        .get(module)
        .and_then(|code_by_arity_by_definition| {
            code_by_arity_by_definition
                .get(definition)
                .map(|located_code_by_arity| located_code_by_arity.contains_key(&arity))
        })
        .unwrap_or(false)
}

pub fn get(module: &Atom, definition: &Definition, arity: Arity) -> Option<LocatedCode> {
    RW_LOCK_LOCATED_CODE_BY_ARITY_BY_DEFINITION_BY_MODULE
        .read()
        .get(module)
        .and_then(|code_by_arity_by_definition| {
            code_by_arity_by_definition
                .get(definition)
                .and_then(|located_code_by_arity| {
                    located_code_by_arity
                        .get(&arity)
                        .map(|located_code| *located_code)
                })
        })
}

pub fn insert(module: Atom, definition: Definition, arity: Arity, located_code: LocatedCode) {
    RW_LOCK_LOCATED_CODE_BY_ARITY_BY_DEFINITION_BY_MODULE
        .write()
        .entry(module)
        .or_insert_with(Default::default)
        .entry(definition)
        .or_insert_with(Default::default)
        .insert(arity, located_code);
}

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    module: Atom,
    function: Atom,
    arity: Arity,
    LocatedCode { code, location }: LocatedCode,
    argument_vec: Vec<Term>,
) -> AllocResult<()> {
    assert_eq!(argument_vec.len(), arity as usize);
    for argument in argument_vec.iter().rev() {
        process.stack_push(*argument)?;
    }

    let frame = Frame::new(module, function, arity, location, code);
    process.place_frame(frame, placement);

    Ok(())
}

lazy_static! {
    static ref RW_LOCK_LOCATED_CODE_BY_ARITY_BY_DEFINITION_BY_MODULE: RwLock<HashMap<Atom, HashMap<Definition, HashMap<Arity, LocatedCode>>>> =
        Default::default();
}
