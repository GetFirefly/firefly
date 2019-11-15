use std::sync::Arc;

use libeir_ir::{BasicType, Block, MatchKind};

use liblumen_alloc::erts::exception::SystemException;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::{Encoded, ExactEq, TypedTerm};

use super::{CallExecutor, OpResult};
use crate::module::ErlangFunction;

pub fn match_op(
    exec: &mut CallExecutor,
    proc: &Arc<Process>,
    fun: &ErlangFunction,
    branches: &[MatchKind],
    block: Block,
) -> std::result::Result<OpResult, SystemException> {
    let reads = fun.fun.block_reads(block);

    let branches_dests = reads[0];

    let unpack_term = exec
        .make_term(proc, fun, reads[1])
        .unwrap()
        .decode()
        .unwrap();

    for (idx, kind) in branches.iter().enumerate() {
        let branch = fun.fun.value_list_get_n(branches_dests, idx).unwrap();

        let branch_args_val = reads[idx + 2];
        let branch_args_len = fun.fun.value_list_length(branch_args_val);

        match kind {
            MatchKind::Value => {
                assert!(branch_args_len == 1);
                let arg = fun.fun.value_list_get_n(branch_args_val, 0).unwrap();
                let rhs = exec.make_term(proc, fun, arg).unwrap();

                if unpack_term.exact_eq(&rhs.decode().unwrap()) {
                    return exec.val_call(proc, fun, branch);
                }
            }
            MatchKind::Type(BasicType::Map) => {
                assert!(branch_args_len == 0);
                if unpack_term.is_map() {
                    return exec.val_call(proc, fun, branch);
                }
            }
            MatchKind::MapItem => {
                assert!(branch_args_len == 1);
                let arg = fun.fun.value_list_get_n(branch_args_val, 0).unwrap();
                let key = exec.make_term(proc, fun, arg).unwrap();

                match unpack_term {
                    TypedTerm::Map(map) => {
                        if let Some(val) = map.get(key) {
                            exec.next_args.push(val);
                            return exec.val_call(proc, fun, branch);
                        }
                    }
                    _ => unreachable!(),
                }
            }
            MatchKind::Tuple(arity) => {
                assert!(branch_args_len == 0);

                match unpack_term {
                    TypedTerm::Tuple(tup) => {
                        if tup.len() == *arity {
                            exec.next_args.extend(tup.iter());
                            return exec.val_call(proc, fun, branch);
                        }
                    }
                    _ => (),
                }
            }
            MatchKind::ListCell => {
                assert!(branch_args_len == 0);

                match unpack_term {
                    TypedTerm::List(cons) => {
                        exec.next_args.push(cons.head);
                        exec.next_args.push(cons.tail);
                        return exec.val_call(proc, fun, branch);
                    }
                    _ => (),
                }
            }
            MatchKind::Wildcard => {
                assert!(branch_args_len == 0);
                return exec.val_call(proc, fun, branch);
            }
            kind => unimplemented!("{:?}", kind),
        }
    }

    panic!()
}
