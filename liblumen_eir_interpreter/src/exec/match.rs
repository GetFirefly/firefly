use std::convert::TryInto;
use std::sync::Arc;

use libeir_ir::{ Block, MatchKind, PrimOpKind, BasicType,
                 BinaryEntrySpecifier, Endianness };

use lumen_runtime::otp::erlang;
use liblumen_alloc::erts::term::{ Term, TypedTerm, Tuple, Atom, Integer, Closure,
                                  AsTerm, atom_unchecked };
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::exception::system;

use crate::module::ErlangFunction;
use super::{ CallExecutor, OpResult };

pub fn match_op(
    exec: &mut CallExecutor,
    proc: &Arc<ProcessControlBlock>,
    fun: &ErlangFunction,
    branches: &[MatchKind],
    block: Block
) -> std::result::Result<OpResult, system::Exception>
{
    let reads = fun.fun.block_reads(block);

    let branches_prim = fun.fun.value_primop(reads[0]).unwrap();
    assert!(fun.fun.primop_kind(branches_prim) == &PrimOpKind::ValueList);
    let branches_dests = fun.fun.primop_reads(branches_prim);

    let unpack_term = exec.make_term(proc, fun, reads[1]).unwrap();

    for (idx, (kind, branch)) in branches.iter()
        .zip(branches_dests.iter()).enumerate()
    {
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
            MatchKind::Wildcard => {
                assert!(branch_args.len() == 0);
                return exec.val_call(proc, fun, *branch);
            }
            kind => unimplemented!("{:?}", kind),
        }
    }

    panic!()
}
