use std::sync::Arc;

use libeir_ir::{BasicType, Block, MatchKind, PrimOpKind};

use liblumen_alloc::erts::exception::system;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::TypedTerm;

use super::{CallExecutor, OpResult};
use crate::module::ErlangFunction;

pub fn match_op(
    exec: &mut CallExecutor,
    proc: &Arc<Process>,
    fun: &ErlangFunction,
    branches: &[MatchKind],
    block: Block,
) -> std::result::Result<OpResult, system::Exception> {
    let reads = fun.fun.block_reads(block);

    let branches_prim = fun.fun.value_primop(reads[0]).unwrap();
    assert!(fun.fun.primop_kind(branches_prim) == &PrimOpKind::ValueList);
    let branches_dests = fun.fun.primop_reads(branches_prim);

    let unpack_term = exec.make_term(proc, fun, reads[1]).unwrap();

    for (idx, (kind, branch)) in branches.iter().zip(branches_dests.iter()).enumerate() {
        let branch_arg_prim = fun.fun.value_primop(reads[idx + 2]).unwrap();
        assert!(fun.fun.primop_kind(branch_arg_prim) == &PrimOpKind::ValueList);
        let branch_args = fun.fun.primop_reads(branch_arg_prim);

        match kind {
            MatchKind::Value => {
                assert!(branch_args.len() == 1);
                let rhs = exec.make_term(proc, fun, branch_args[0]).unwrap();

                if unpack_term.exactly_eq(&rhs) {
                    return exec.val_call(proc, fun, *branch);
                }
            }
            MatchKind::Type(BasicType::Map) => {
                assert!(branch_args.len() == 0);
                if unpack_term.is_map() {
                    return exec.val_call(proc, fun, *branch);
                }
            }
            MatchKind::MapItem => {
                assert!(branch_args.len() == 1);
                let key = exec.make_term(proc, fun, branch_args[0]).unwrap();

                match unpack_term.to_typed_term().unwrap() {
                    TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                        TypedTerm::Map(map) => {
                            if let Some(val) = map.get(key) {
                                exec.next_args.push(val);
                                return exec.val_call(proc, fun, *branch);
                            }
                        }
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                }
            }
            MatchKind::Wildcard => {
                assert!(branch_args.len() == 0);
                return exec.val_call(proc, fun, *branch);
            }
            kind => unimplemented!("{:?}", kind),
        }
    }

    panic!()
}
