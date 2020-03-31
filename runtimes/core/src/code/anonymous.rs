use hashbrown::hash_map::HashMap;
use lazy_static::lazy_static;

use liblumen_core::locks::RwLock;

use liblumen_alloc::erts::process::code::Code;
use liblumen_alloc::erts::term::closure::{Index, OldUnique, Unique};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Arity;

pub fn get(
    module: &Atom,
    index: &Index,
    old_unique: &OldUnique,
    unique: &Unique,
    arity: &Arity,
) -> Option<Code> {
    RW_LOCK_CODE_BY_ARITY_BY_UNIQUE_BY_OLD_UNIQUE_BY_INDEX_BY_MODULE
        .read()
        .get(module)
        .and_then(|code_by_arity_by_unique_by_old_unique_by_index| {
            code_by_arity_by_unique_by_old_unique_by_index
                .get(index)
                .and_then(|code_by_arity_by_unique_by_old_unique| {
                    code_by_arity_by_unique_by_old_unique
                        .get(old_unique)
                        .and_then(|code_by_arity_by_unique| {
                            code_by_arity_by_unique
                                .get(unique)
                                .and_then(|code_by_arity| {
                                    code_by_arity.get(arity).map(|code| *code)
                                })
                        })
                })
        })
}

pub fn insert(
    module: Atom,
    index: Index,
    old_unique: OldUnique,
    unique: Unique,
    arity: Arity,
    code: Code,
) {
    RW_LOCK_CODE_BY_ARITY_BY_UNIQUE_BY_OLD_UNIQUE_BY_INDEX_BY_MODULE
        .write()
        .entry(module)
        .or_insert_with(Default::default)
        .entry(index)
        .or_insert_with(Default::default)
        .entry(old_unique)
        .or_insert_with(Default::default)
        .entry(unique)
        .or_insert_with(Default::default)
        .insert(arity, code);
}

lazy_static! {
    static ref RW_LOCK_CODE_BY_ARITY_BY_UNIQUE_BY_OLD_UNIQUE_BY_INDEX_BY_MODULE: RwLock<HashMap<Atom, HashMap<Index, HashMap<OldUnique, HashMap<Unique, HashMap<Arity, Code>>>>>> =
        Default::default();
}
