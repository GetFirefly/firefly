// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::Term;(

use lumen_runtime_macros::native_implemented_function;

use crate::otp::lists::get_by_term_one_based_index_key;

#[native_implemented_function(keyfind/3)]
pub fn native(key: Term, one_based_index: Term, tuple_list: Term) -> exception::Result {
    get_by_term_one_based_index_key(tuple_list, one_based_index, key).map(|option| match option {
        Some(found) => found,
        None => false.into(),
    })
}
