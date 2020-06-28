mod without_native;

use std::sync::Arc;

use proptest::prop_oneof;
use proptest::strategy::{BoxedStrategy, Just, Strategy};

use liblumen_alloc::erts::term::prelude::{Atom, Term};
use liblumen_alloc::erts::Process;

use crate::erlang;
use crate::test::strategy;

use super::{arity_u8, module_atom};

pub fn function() -> BoxedStrategy<Atom> {
    strategy::atom()
}

pub fn with_arity(arc_process: Arc<Process>, arity: u8) -> BoxedStrategy<Term> {
    match arity {
        0 => prop_oneof![
            Just(
                arc_process
                    .export_closure(
                        erlang::module(),
                        erlang::self_0::function(),
                        erlang::self_0::ARITY,
                        Some(erlang::self_0::native as _)
                    )
                    .unwrap()
            ),
            without_native::with_arity(arc_process, arity)
        ]
        .boxed(),
        1 => prop_oneof![
            Just(
                arc_process
                    .export_closure(
                        erlang::module(),
                        erlang::number_or_badarith_1::function(),
                        erlang::number_or_badarith_1::ARITY,
                        Some(erlang::number_or_badarith_1::native as _)
                    )
                    .unwrap()
            ),
            without_native::with_arity(arc_process, arity)
        ]
        .boxed(),
        _ => without_native::with_arity(arc_process, arity),
    }
}

pub fn without_native(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (module_atom(), function(), arity_u8())
        .prop_map(move |(module, function, arity)| {
            arc_process
                .export_closure(module, function, arity, None)
                .unwrap()
        })
        .boxed()
}

pub fn with_native(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (0..=1)
        .prop_map(move |arity| {
            let module = erlang::module();

            // MUST be functions in symbol table passed to `runtime::test::once` in
            // `test::process::init`.
            match arity {
                0 => arc_process
                    .export_closure(
                        module,
                        erlang::self_0::function(),
                        erlang::self_0::ARITY,
                        Some(erlang::self_0::native as _),
                    )
                    .unwrap(),
                1 => arc_process
                    .export_closure(
                        module,
                        erlang::number_or_badarith_1::function(),
                        erlang::number_or_badarith_1::ARITY,
                        Some(erlang::number_or_badarith_1::native as _),
                    )
                    .unwrap(),
                _ => unreachable!("arity {}", arity),
            }
        })
        .boxed()
}
