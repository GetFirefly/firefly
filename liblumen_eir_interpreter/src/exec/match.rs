use libeir_ir::{ Block, MatchKind, BasicType, BinaryEntrySpecifier, Endianness };

use libeir_util::binary::{ BitCarrier };
use libeir_util::binary::{ BitVec, BitSlice, carrier_to_integer, Endian };

use crate::{ Term };
use crate::module::ErlangFunction;
use crate::term::ErlExactEq;

use super::{ CallExecutor, TermCall };

pub fn match_op(
    exec: &mut CallExecutor,
    fun: &ErlangFunction,
    branches: &[MatchKind],
    block: Block
) -> TermCall
{
    let reads = fun.fun.block_reads(block);

    let branches_elems = Term::as_value_list(
        &exec.make_term(fun, reads[0]));

    let unpack_term = exec.make_term(fun, reads[1]);

    //println!("MATCH START {:?}", unpack_term);
    for (idx, kind) in branches.iter().enumerate() {
        let branch_args = Term::as_value_list(
            &exec.make_term(fun, reads[idx + 2]));
        //println!("MATCH KIND {:?} {:?}", kind, branch_args);
        match kind {
            MatchKind::Value => {
                assert!(branch_args.len() == 1);
                if unpack_term.erl_exact_eq(&*branch_args[0]) {
                    return TermCall {
                        fun: branches_elems[idx].clone(),
                        args: vec![],
                    };
                }
            }
            MatchKind::ListCell => {
                assert!(branch_args.len() == 0);
                match &*unpack_term {
                    Term::ListCell(head, tail) => {
                        return TermCall {
                            fun: branches_elems[idx].clone(),
                            args: vec![head.clone(), tail.clone()],
                        };
                    }
                    _ => (),
                }
            }
            MatchKind::Tuple(len) => {
                assert!(branch_args.len() == 0);
                match &*unpack_term {
                    Term::Tuple(elems) if elems.len() == *len => {
                        return TermCall {
                            fun: branches_elems[idx].clone(),
                            args: elems.clone(),
                        };
                    }
                    _ => (),
                }
            }
            MatchKind::Type(BasicType::Map) => {
                assert!(branch_args.len() == 0);
                match &*unpack_term {
                    Term::Map(_) => {
                        return TermCall {
                            fun: branches_elems[idx].clone(),
                            args: vec![],
                        };
                    }
                    _ => (),
                }
            }
            MatchKind::MapItem => {
                assert!(branch_args.len() == 1);
                match &*unpack_term {
                    Term::Map(map) => {
                        if let Some(v) = map.get(&branch_args[0]) {
                            return TermCall {
                                fun: branches_elems[idx].clone(),
                                args: vec![v.clone()],
                            };
                        }
                    }
                    _ => unreachable!(),
                }
            }
            MatchKind::Binary(BinaryEntrySpecifier::Integer {
                unit, endianness, signed
            }) => {
                let size = branch_args[0].as_usize().unwrap();
                let bit_len = (*unit as usize) * size;

                let ret = match &*unpack_term {
                    Term::Binary(bin) => {
                        if (bin.len() * 8) < bit_len { continue; }

                        let int_slice = BitSlice::with_offset_length(
                            &**bin, 0, bit_len);
                        let endian = match *endianness {
                            Endianness::Big => Endian::Big,
                            Endianness::Little => Endian::Little,
                            Endianness::Native => Endian::Big,
                        };
                        let int = carrier_to_integer(int_slice, *signed, endian);

                        TermCall {
                            fun: branches_elems[idx].clone(),
                            args: vec![
                                Term::Integer(int).into(),
                                Term::BinarySlice {
                                    buf: bin.clone(),
                                    bit_offset: bit_len,
                                    bit_length: bin.bit_len() - bit_len,
                                }.into(),
                            ],
                        }
                    },
                    Term::BinarySlice { buf, bit_offset, bit_length } => {
                        if *bit_length < bit_len { continue; }

                        let int_slice = BitSlice::with_offset_length(
                            &**buf, *bit_offset, bit_len);
                        let endian = match *endianness {
                            Endianness::Big => Endian::Big,
                            Endianness::Little => Endian::Little,
                            Endianness::Native => Endian::Big,
                        };
                        let int = carrier_to_integer(int_slice, *signed, endian);

                        TermCall {
                            fun: branches_elems[idx].clone(),
                            args: vec![
                                Term::Integer(int).into(),
                                Term::BinarySlice {
                                    buf: buf.clone(),
                                    bit_offset: *bit_offset + bit_len,
                                    bit_length: *bit_length - bit_len,
                                }.into(),
                            ],
                        }
                    },
                    _ => continue,
                };
                return ret;
            }
            MatchKind::Binary(BinaryEntrySpecifier::Bytes { unit: 8 }) => {
                match &*unpack_term {
                    Term::Binary(bin) => {
                        if bin.bit_len() % 8 != 0 { continue; }

                        return TermCall {
                            fun: branches_elems[idx].clone(),
                            args: vec![
                                unpack_term.clone(),
                                Term::Binary(BitVec::new().into()).into(),
                            ],
                        };
                    }
                    Term::BinarySlice { bit_length, .. } => {
                        if *bit_length % 8 != 0 { continue; }

                        return TermCall {
                            fun: branches_elems[idx].clone(),
                            args: vec![
                                unpack_term.clone(),
                                Term::Binary(BitVec::new().into()).into(),
                            ],
                        };
                    }
                    _ => (),
                }
            }
            MatchKind::Wildcard => {
                assert!(branch_args.len() == 0);
                return TermCall {
                    fun: branches_elems[idx].clone(),
                    args: vec![],
                };
            }
            kind => unimplemented!("{:?}", kind),
        }
    }

    panic!()
}
