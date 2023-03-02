use core::fmt::{self, Write};

use firefly_binary::BinaryEntrySpecifier;

use super::ops::BsMatchSkipType;
use super::*;

pub fn write<A, T>(w: &mut dyn Write, module: &ByteCode<A, T>) -> fmt::Result
where
    A: Atom,
    T: AtomTable<Atom = A>,
{
    if module.code.is_empty() {
        return Ok(());
    }

    let mut current_function = module.function_by_ip(5);
    w.write_fmt(format_args!(
        "fun {}: # offset={}",
        current_function.mfa().unwrap(),
        3
    ))?;
    for (ip, op) in module.code.iter().enumerate().skip(5) {
        let f = module.function_by_ip(ip);
        if f.offset() != current_function.offset() {
            current_function = f;
            w.write_fmt(format_args!(
                "\n\nfun {}: # offset={}",
                current_function.mfa().unwrap(),
                f.offset().unwrap()
            ))?;
        }

        let function_offset = current_function.offset().unwrap();
        write_opcode(w, module, ip - function_offset, op)?;
    }

    Ok(())
}

fn write_opcode<A, T>(
    w: &mut dyn Write,
    module: &ByteCode<A, T>,
    offset: usize,
    op: &Opcode<A>,
) -> fmt::Result
where
    A: Atom,
    T: AtomTable<Atom = A>,
{
    w.write_fmt(format_args!("\n  {:<4}| ", offset))?;
    match op {
        Opcode::Nop(_) => w.write_str("nop"),
        Opcode::Mov(op) => w.write_fmt(format_args!("mov ${}, ${}", op.dest, op.src)),
        Opcode::Cmov(op) => {
            w.write_fmt(format_args!("cmov ${}, ${}, ${}", op.cond, op.dest, op.src))
        }
        Opcode::Ret(op) => w.write_fmt(format_args!("ret ${}", op.reg)),
        Opcode::Br(op) => w.write_fmt(format_args!(
            "br {}",
            offset.checked_add_signed(op.offset as isize).unwrap()
        )),
        Opcode::Brz(op) => w.write_fmt(format_args!(
            "brz ${}, {}",
            op.reg,
            offset.checked_add_signed(op.offset as isize).unwrap()
        )),
        Opcode::Brnz(op) => w.write_fmt(format_args!(
            "brnz ${}, {}",
            op.reg,
            offset.checked_add_signed(op.offset as isize).unwrap()
        )),
        Opcode::JumpTable(op) => w.write_fmt(format_args!("jt ${} # len={}", op.reg, op.len)),
        Opcode::JumpTableEntry(op) => w.write_fmt(format_args!(
            "jt.entry {}, {}",
            op.imm,
            offset.checked_add_signed(op.offset as isize).unwrap()
        )),
        Opcode::Call(op) => {
            let f = module.function_by_ip(op.offset);
            let mfa = f.mfa().unwrap();
            w.write_fmt(format_args!(
                "call ${}, {} # offset={}",
                op.dest, mfa, op.offset
            ))
        }
        Opcode::CallApply2(op) => w.write_fmt(format_args!(
            "call.apply2 ${}, ${}, ${}",
            op.dest, op.callee, op.argv
        )),
        Opcode::CallApply3(op) => w.write_fmt(format_args!(
            "call.apply3 ${}, ${}, ${}, ${}",
            op.dest, op.module, op.function, op.argv
        )),
        Opcode::CallNative(op) => w.write_fmt(format_args!(
            "call.native.{} ${}, {:p}",
            op.arity, op.dest, op.callee
        )),
        Opcode::CallStatic(op) => {
            let f = module.function_by_id(op.callee);
            match f {
                Function::Bytecode {
                    frame_size,
                    is_nif,
                    mfa,
                    ..
                } => w.write_fmt(format_args!(
                    "call.static ${}, {} # frame_size={} is_nif={}",
                    op.dest, mfa, frame_size, is_nif
                )),
                Function::Bif { mfa, .. } => w.write_fmt(format_args!(
                    "call.static ${}, {} # is_bif=true",
                    op.dest, mfa
                )),
                Function::Native { name, arity, .. } => w.write_fmt(format_args!(
                    "call.static ${}, {}/{} # is_nif=true",
                    op.dest, name, arity
                )),
            }
        }
        Opcode::CallIndirect(op) => w.write_fmt(format_args!(
            "call.indirect.{} ${}, ${}",
            op.arity, op.dest, op.callee
        )),
        Opcode::Enter(op) => {
            let f = module.function_by_ip(op.offset);
            let mfa = f.mfa().unwrap();
            w.write_fmt(format_args!("enter {} # offset={}", mfa, op.offset))
        }
        Opcode::EnterApply2(op) => {
            w.write_fmt(format_args!("enter.apply2 ${}, ${}", op.callee, op.argv))
        }
        Opcode::EnterApply3(op) => w.write_fmt(format_args!(
            "enter.apply3 ${}, ${}, ${}",
            op.module, op.function, op.argv
        )),
        Opcode::EnterNative(op) => {
            w.write_fmt(format_args!("enter.native.{} {:p}", op.arity, op.callee))
        }
        Opcode::EnterStatic(op) => {
            let f = module.function_by_id(op.callee);
            match f {
                Function::Bytecode {
                    frame_size,
                    is_nif,
                    mfa,
                    ..
                } => w.write_fmt(format_args!(
                    "enter.static {} # frame_size={} is_nif={}",
                    mfa, frame_size, is_nif
                )),
                Function::Bif { mfa, .. } => {
                    w.write_fmt(format_args!("enter.static {} # is_bif=true", mfa))
                }
                Function::Native { name, arity, .. } => w.write_fmt(format_args!(
                    "enter.static {}/{} # is_nif=true",
                    name, arity
                )),
            }
        }
        Opcode::EnterIndirect(op) => {
            w.write_fmt(format_args!("enter.indirect.{} ${}", op.arity, op.callee))
        }
        Opcode::IsAtom(op) => w.write_fmt(format_args!("is_atom ${}, ${}", op.dest, op.value)),
        Opcode::IsBool(op) => w.write_fmt(format_args!("is_bool ${}, ${}", op.dest, op.value)),
        Opcode::IsNil(op) => w.write_fmt(format_args!("is_nil ${}, ${}", op.dest, op.value)),
        Opcode::IsTuple(op) => match op.arity {
            None => w.write_fmt(format_args!("is_tuple ${}, ${}", op.dest, op.value)),
            Some(n) => w.write_fmt(format_args!(
                "is_tuple.{} ${}, ${}",
                n.get(),
                op.dest,
                op.value
            )),
        },
        Opcode::IsTupleFetchArity(op) => w.write_fmt(format_args!(
            "is_tuple_fetch_arity ${}, ${}, ${}",
            op.dest, op.value, op.arity
        )),
        Opcode::IsMap(op) => w.write_fmt(format_args!("is_map ${}, ${}", op.dest, op.value)),
        Opcode::IsCons(op) => w.write_fmt(format_args!("is_cons ${}, ${}", op.dest, op.value)),
        Opcode::IsList(op) => w.write_fmt(format_args!("is_list ${}, ${}", op.dest, op.value)),
        Opcode::IsInt(op) => w.write_fmt(format_args!("is_int ${}, ${}", op.dest, op.value)),
        Opcode::IsFloat(op) => w.write_fmt(format_args!("is_float ${}, ${}", op.dest, op.value)),
        Opcode::IsNumber(op) => w.write_fmt(format_args!("is_number ${}, ${}", op.dest, op.value)),
        Opcode::IsPid(op) => w.write_fmt(format_args!("is_pid ${}, ${}", op.dest, op.value)),
        Opcode::IsRef(op) => w.write_fmt(format_args!("is_ref ${}, ${}", op.dest, op.value)),
        Opcode::IsPort(op) => w.write_fmt(format_args!("is_port ${}, ${}", op.dest, op.value)),
        Opcode::IsBinary(op) => match op.unit {
            8 => w.write_fmt(format_args!("is_binary ${}, ${}", op.dest, op.value)),
            _ => w.write_fmt(format_args!("is_bitstring ${}, ${}", op.dest, op.value)),
        },
        Opcode::IsFunction(op) => {
            w.write_fmt(format_args!("is_function ${}, ${}", op.dest, op.value))
        }
        Opcode::LoadNil(op) => w.write_fmt(format_args!("load_nil ${}", op.dest)),
        Opcode::LoadBool(op) => w.write_fmt(format_args!("load_bool ${}, {}", op.dest, op.value)),
        Opcode::LoadAtom(op) => w.write_fmt(format_args!("load_atom ${}, {}", op.dest, op.value)),
        Opcode::LoadInt(op) => w.write_fmt(format_args!("load_int ${}, {}", op.dest, op.value)),
        Opcode::LoadBig(op) => w.write_fmt(format_args!("load_big ${}, {}", op.dest, &op.value)),
        Opcode::LoadFloat(op) => w.write_fmt(format_args!("load_float ${}, {}", op.dest, op.value)),
        Opcode::LoadBinary(op) => {
            w.write_fmt(format_args!("load_binary ${}, {}", op.dest, unsafe {
                &*op.value
            }))
        }
        Opcode::LoadBitstring(op) => {
            w.write_fmt(format_args!("load_bitstring ${}, {}", op.dest, unsafe {
                &*op.value
            }))
        }
        Opcode::Not(op) => w.write_fmt(format_args!("not ${}, ${}", op.dest, op.cond)),
        Opcode::And(op) => w.write_fmt(format_args!("and ${}, ${}, ${}", op.dest, op.lhs, op.rhs)),
        Opcode::AndAlso(op) => w.write_fmt(format_args!(
            "andalso ${}, ${}, ${}",
            op.dest, op.lhs, op.rhs
        )),
        Opcode::Or(op) => w.write_fmt(format_args!("or ${}, ${}, ${}", op.dest, op.lhs, op.rhs)),
        Opcode::OrElse(op) => w.write_fmt(format_args!(
            "orelse ${}, ${}, ${}",
            op.dest, op.lhs, op.rhs
        )),
        Opcode::Xor(op) => w.write_fmt(format_args!("xor ${}, ${}, ${}", op.dest, op.lhs, op.rhs)),
        Opcode::Bnot(op) => w.write_fmt(format_args!("bnot ${}, ${}", op.dest, op.rhs)),
        Opcode::Band(op) => {
            w.write_fmt(format_args!("band ${}, ${}, ${}", op.dest, op.lhs, op.rhs))
        }
        Opcode::Bor(op) => w.write_fmt(format_args!("bor ${}, ${}, ${}", op.dest, op.lhs, op.rhs)),
        Opcode::Bxor(op) => {
            w.write_fmt(format_args!("bxor ${}, ${}, ${}", op.dest, op.lhs, op.rhs))
        }
        Opcode::Bsl(op) => w.write_fmt(format_args!(
            "bsl ${}, ${}, ${}",
            op.dest, op.value, op.shift
        )),
        Opcode::Bsr(op) => w.write_fmt(format_args!(
            "bsr ${}, ${}, ${}",
            op.dest, op.value, op.shift
        )),
        Opcode::Div(op) => w.write_fmt(format_args!(
            "div ${}, ${}, ${}",
            op.dest, op.value, op.divisor
        )),
        Opcode::Rem(op) => w.write_fmt(format_args!(
            "rem ${}, ${}, ${}",
            op.dest, op.value, op.divisor
        )),
        Opcode::Neg(op) => w.write_fmt(format_args!("neg ${}, ${}", op.dest, op.rhs)),
        Opcode::Add(op) => w.write_fmt(format_args!("add ${}, ${}, ${}", op.dest, op.lhs, op.rhs)),
        Opcode::Sub(op) => w.write_fmt(format_args!("sub ${}, ${}, ${}", op.dest, op.lhs, op.rhs)),
        Opcode::Mul(op) => w.write_fmt(format_args!("mul ${}, ${}, ${}", op.dest, op.lhs, op.rhs)),
        Opcode::Divide(op) => {
            w.write_fmt(format_args!("fdiv ${}, ${}, ${}", op.dest, op.lhs, op.rhs))
        }
        Opcode::ListAppend(op) => w.write_fmt(format_args!(
            "list_append ${}, ${}, ${}",
            op.dest, op.list, op.rhs
        )),
        Opcode::ListRemove(op) => w.write_fmt(format_args!(
            "list_remove ${}, ${}, ${}",
            op.dest, op.list, op.rhs
        )),
        Opcode::Eq(op) => {
            if op.strict {
                w.write_fmt(format_args!(
                    "eq.strict ${}, ${}, ${}",
                    op.dest, op.lhs, op.rhs
                ))
            } else {
                w.write_fmt(format_args!("eq ${}, ${}, ${}", op.dest, op.lhs, op.rhs))
            }
        }
        Opcode::Neq(op) => {
            if op.strict {
                w.write_fmt(format_args!(
                    "neq.strict ${}, ${}, ${}",
                    op.dest, op.lhs, op.rhs
                ))
            } else {
                w.write_fmt(format_args!("neq ${}, ${}, ${}", op.dest, op.lhs, op.rhs))
            }
        }
        Opcode::Gt(op) => w.write_fmt(format_args!("gt ${}, ${}, ${}", op.dest, op.lhs, op.rhs)),
        Opcode::Gte(op) => w.write_fmt(format_args!("gte ${}, ${}, ${}", op.dest, op.lhs, op.rhs)),
        Opcode::Lt(op) => w.write_fmt(format_args!("lt ${}, ${}, ${}", op.dest, op.lhs, op.rhs)),
        Opcode::Lte(op) => w.write_fmt(format_args!("lte ${}, ${}, ${}", op.dest, op.lhs, op.rhs)),
        Opcode::Cons(op) => w.write_fmt(format_args!(
            "cons ${}, ${}, ${}",
            op.dest, op.head, op.tail
        )),
        Opcode::Split(op) => {
            w.write_fmt(format_args!("split ${}, ${}, ${}", op.hd, op.tl, op.list))
        }
        Opcode::Head(op) => w.write_fmt(format_args!("hd ${}, ${}", op.dest, op.list)),
        Opcode::Tail(op) => w.write_fmt(format_args!("tl ${}, ${}", op.dest, op.list)),
        Opcode::Closure(op) => {
            let f = module.function_by_id(op.function);
            match f {
                Function::Bytecode {
                    frame_size,
                    is_nif,
                    mfa,
                    ..
                } => w.write_fmt(format_args!(
                    "closure.{} ${}, {} # frame_size={} is_nif={}",
                    op.arity, op.dest, mfa, frame_size, is_nif
                )),
                Function::Bif { mfa, .. } => w.write_fmt(format_args!(
                    "closure.{} ${}, {} # is_bif=true",
                    op.arity, op.dest, mfa
                )),
                Function::Native { name, .. } => w.write_fmt(format_args!(
                    "closure.{} ${}, {} # is_nif=true",
                    op.arity, op.dest, name
                )),
            }
        }
        Opcode::UnpackEnv(op) => w.write_fmt(format_args!(
            "unpack_env ${}, ${}[{}]",
            op.dest, op.fun, op.index
        )),
        Opcode::Tuple(op) => w.write_fmt(format_args!("tuple.{} ${}", op.arity, op.dest)),
        Opcode::TupleWithCapacity(op) => w.write_fmt(format_args!(
            "tuple_with_capacity.{} ${}",
            op.arity, op.dest
        )),
        Opcode::TupleArity(op) => {
            w.write_fmt(format_args!("tuple_arity ${}, ${}", op.dest, op.tuple))
        }
        Opcode::GetElement(op) => w.write_fmt(format_args!(
            "get_element ${}, ${}[{}]",
            op.dest, op.tuple, op.index
        )),
        Opcode::SetElement(op) => w.write_fmt(format_args!(
            "set_element ${}, ${}[{}], ${}",
            op.dest, op.tuple, op.index, op.value
        )),
        Opcode::SetElementMut(op) => w.write_fmt(format_args!(
            "set_element.mut ${}[{}], ${}",
            op.tuple, op.index, op.value
        )),
        Opcode::Map(op) => w.write_fmt(format_args!("map.{} ${}", op.dest, op.capacity)),
        Opcode::MapPut(op) => w.write_fmt(format_args!(
            "map_put ${}, ${}[${}], ${}",
            op.dest, op.map, op.key, op.value
        )),
        Opcode::MapPutMut(op) => w.write_fmt(format_args!(
            "map_put.mut ${}[${}], ${}",
            op.map, op.key, op.value
        )),
        Opcode::MapUpdate(op) => w.write_fmt(format_args!(
            "map_update ${}, ${}[${}], ${}",
            op.dest, op.map, op.key, op.value
        )),
        Opcode::MapUpdateMut(op) => w.write_fmt(format_args!(
            "map_update.mut ${}[${}], ${}",
            op.map, op.key, op.value
        )),
        Opcode::MapExtendPut(op) => {
            assert!(!op.pairs.is_empty());
            assert!(op.pairs.len() % 2 == 0);
            w.write_fmt(format_args!("map_extend.put ${}", op.map))?;
            for [k, v] in unsafe { op.pairs.as_slice().as_chunks_unchecked() } {
                w.write_fmt(format_args!(", (${}, ${})", k, v))?;
            }
            Ok(())
        }
        Opcode::MapExtendUpdate(op) => {
            assert!(!op.pairs.is_empty());
            assert!(op.pairs.len() % 2 == 0);
            w.write_fmt(format_args!("map_extend.update ${}", op.map))?;
            for [k, v] in unsafe { op.pairs.as_slice().as_chunks_unchecked() } {
                w.write_fmt(format_args!(", (${}, ${})", k, v))?;
            }
            Ok(())
        }
        Opcode::MapTryGet(op) => w.write_fmt(format_args!(
            "map_try_get ${}, ${}, ${}[${}]",
            op.is_err, op.value, op.map, op.key
        )),
        Opcode::Catch(op) => w.write_fmt(format_args!("catch ${}", op.cp)),
        Opcode::EndCatch(_) => w.write_str("end_catch"),
        Opcode::LandingPad(op) => w.write_fmt(format_args!(
            "landing_pad ${}, ${}, ${}, {}",
            op.kind,
            op.reason,
            op.trace,
            offset.checked_add_signed(op.offset as isize).unwrap()
        )),
        Opcode::StackTrace(op) => w.write_fmt(format_args!("stacktrace ${}", op.dest)),
        Opcode::Send(op) => w.write_fmt(format_args!("send ${}, ${}", op.recipient, op.message)),
        Opcode::RecvPeek(op) => {
            w.write_fmt(format_args!("recv_peek ${}, ${}", op.available, op.message))
        }
        Opcode::RecvNext(_) => w.write_str("recv_next"),
        Opcode::RecvWait(op) => {
            w.write_fmt(format_args!("recv_wait ${}, ${}", op.dest, op.timeout))
        }
        Opcode::RecvTimeout(op) => w.write_fmt(format_args!("recv_timeout ${}", op.dest)),
        Opcode::RecvPop(_) => w.write_str("recv_pop"),
        Opcode::Await(_) => w.write_str("await"),
        Opcode::Trap(_) => w.write_str("trap"),
        Opcode::Yield(_) => w.write_str("yield"),
        Opcode::GarbageCollect(op) => {
            if op.fullsweep {
                w.write_str("gc.fullsweep")
            } else {
                w.write_str("gc")
            }
        }
        Opcode::NormalExit(_) => w.write_str("normal_exit"),
        Opcode::ContinueExit(_) => w.write_str("continue_exit"),
        Opcode::Exit1(op) => w.write_fmt(format_args!("exit1 ${}", op.reason)),
        Opcode::Exit2(op) => w.write_fmt(format_args!(
            "exit2 ${}, ${}, ${}",
            op.dest, op.pid, op.reason
        )),
        Opcode::Raise(op) => match op.trace {
            Some(t) => w.write_fmt(format_args!(
                "raise ${}, ${}, ${}, ${}",
                op.dest, op.kind, op.reason, t
            )),
            None => w.write_fmt(format_args!(
                "raise ${}, ${}, ${}",
                op.dest, op.kind, op.reason
            )),
        },
        Opcode::Error1(op) => w.write_fmt(format_args!("error1 ${}", op.reason)),
        Opcode::Throw1(op) => w.write_fmt(format_args!("throw1 ${}", op.reason)),
        Opcode::Halt(op) => w.write_fmt(format_args!("halt ${}, ${}", op.status, op.options)),
        Opcode::BsInit(op) => w.write_fmt(format_args!("bs_init ${}", op.dest)),
        Opcode::BsPush(op) => {
            match op.spec {
                BinaryEntrySpecifier::Integer {
                    signed: true,
                    endianness,
                    unit,
                } => w.write_fmt(format_args!(
                    "bs_push.int.signed.{}.{} ${}, ${}, ${}",
                    endianness, unit, op.dest, op.builder, op.value
                ))?,
                BinaryEntrySpecifier::Integer {
                    endianness, unit, ..
                } => w.write_fmt(format_args!(
                    "bs_push.int.unsigned.{}.{} ${}, ${}, ${}",
                    endianness, unit, op.dest, op.builder, op.value
                ))?,
                BinaryEntrySpecifier::Float { endianness, unit } => w.write_fmt(format_args!(
                    "bs_push.float.{}.{} ${}, ${}, ${}",
                    endianness, unit, op.dest, op.builder, op.value
                ))?,
                BinaryEntrySpecifier::Binary { unit } => w.write_fmt(format_args!(
                    "bs_push.binary.{} ${}, ${}, ${}",
                    unit, op.dest, op.builder, op.value
                ))?,
                BinaryEntrySpecifier::Utf8 => w.write_fmt(format_args!(
                    "bs_push.utf8 ${}, ${}, ${}",
                    op.dest, op.builder, op.value
                ))?,
                BinaryEntrySpecifier::Utf16 { endianness } => w.write_fmt(format_args!(
                    "bs_push.utf16.{} ${}, ${}, ${}",
                    endianness, op.dest, op.builder, op.value
                ))?,
                BinaryEntrySpecifier::Utf32 { endianness } => w.write_fmt(format_args!(
                    "bs_push.utf32.{} ${}, ${}, ${}",
                    endianness, op.dest, op.builder, op.value
                ))?,
            }
            match op.size {
                None => Ok(()),
                Some(sz) => w.write_fmt(format_args!(", ${}", sz)),
            }
        }
        Opcode::BsFinish(op) => {
            w.write_fmt(format_args!("bs_finish ${}, ${}", op.dest, op.builder))
        }
        Opcode::BsMatchStart(op) => w.write_fmt(format_args!(
            "bs_match_start ${}, ${}, ${}",
            op.is_err, op.context, op.bin
        )),
        Opcode::BsMatch(op) => {
            match op.spec {
                BinaryEntrySpecifier::Integer {
                    signed: true,
                    endianness,
                    unit,
                } => w.write_fmt(format_args!(
                    "bs_match.int.signed.{}.{} ${}, ${}, ${}, ${}",
                    endianness, unit, op.is_err, op.value, op.next, op.context
                ))?,
                BinaryEntrySpecifier::Integer {
                    endianness, unit, ..
                } => w.write_fmt(format_args!(
                    "bs_match.int.unsigned.{}.{} ${}, ${}, ${}, ${}",
                    endianness, unit, op.is_err, op.value, op.next, op.context
                ))?,
                BinaryEntrySpecifier::Float { endianness, unit } => w.write_fmt(format_args!(
                    "bs_match.float.{}.{} ${}, ${}, ${}, ${}",
                    endianness, unit, op.is_err, op.value, op.next, op.context
                ))?,
                BinaryEntrySpecifier::Binary { unit } => {
                    if unit == 8 {
                        w.write_fmt(format_args!(
                            "bs_match.binary ${}, ${}, ${}, ${}",
                            op.is_err, op.value, op.next, op.context
                        ))?
                    } else {
                        w.write_fmt(format_args!(
                            "bs_match.bitstring.{} ${}, ${}, ${}, ${}",
                            unit, op.is_err, op.value, op.next, op.context
                        ))?
                    }
                }
                BinaryEntrySpecifier::Utf8 => w.write_fmt(format_args!(
                    "bs_match.utf8 ${}, ${}, ${}, ${}",
                    op.is_err, op.value, op.next, op.context
                ))?,
                BinaryEntrySpecifier::Utf16 { endianness } => w.write_fmt(format_args!(
                    "bs_match.utf16.{} ${}, ${}, ${}, ${}",
                    endianness, op.is_err, op.value, op.next, op.context
                ))?,
                BinaryEntrySpecifier::Utf32 { endianness } => w.write_fmt(format_args!(
                    "bs_match.utf32.{} ${}, ${}, ${}, ${}",
                    endianness, op.is_err, op.value, op.next, op.context
                ))?,
            };
            if let Some(sz) = op.size {
                w.write_fmt(format_args!(", ${}", sz))
            } else {
                Ok(())
            }
        }
        Opcode::BsMatchSkip(op) => match op.ty {
            BsMatchSkipType::BigUnsigned => w.write_fmt(format_args!(
                "bs_match_skip.unsigned.big.{} ${}, ${}, ${}, ${}",
                op.unit, op.is_err, op.next, op.context, op.size
            )),
            BsMatchSkipType::BigSigned => w.write_fmt(format_args!(
                "bs_match_skip.signed.big.{} ${}, ${}, ${}, ${}",
                op.unit, op.is_err, op.next, op.context, op.size
            )),
            BsMatchSkipType::LittleUnsigned => w.write_fmt(format_args!(
                "bs_match_skip.unsigned.little.{} ${}, ${}, ${}, ${}",
                op.unit, op.is_err, op.next, op.context, op.size
            )),
            BsMatchSkipType::LittleSigned => w.write_fmt(format_args!(
                "bs_match_skip.signed.little.{} ${}, ${}, ${}, ${}",
                op.unit, op.is_err, op.next, op.context, op.size
            )),
            BsMatchSkipType::NativeUnsigned => w.write_fmt(format_args!(
                "bs_match_skip.unsigned.native.{} ${}, ${}, ${}, ${}",
                op.unit, op.is_err, op.next, op.context, op.size
            )),
            BsMatchSkipType::NativeSigned => w.write_fmt(format_args!(
                "bs_match_skip.signed.native.{} ${}, ${}, ${}, ${}",
                op.unit, op.is_err, op.next, op.context, op.size
            )),
        },
        Opcode::BsTestTail(op) => w.write_fmt(format_args!(
            "bs_test_tail.{} ${}, ${}",
            op.size, op.dest, op.context
        )),
        Opcode::FuncInfo(op) => w.write_fmt(format_args!(
            "func_info # id={} arity={} frame_size={}",
            op.id, op.arity, op.frame_size
        )),
        Opcode::Identity(op) => w.write_fmt(format_args!("self ${}", op.dest)),
        Opcode::Spawn2(op) => w.write_fmt(format_args!(
            "spawn2 ${}, ${} # {:?}",
            op.dest, op.fun, op.opts
        )),
        Opcode::Spawn3(op) => {
            let f = module.function_by_id(op.fun);
            let mfa = f.mfa().unwrap();
            w.write_fmt(format_args!(
                "spawn3 ${}, {}, ${} # {:?}",
                op.dest, mfa, op.args, op.opts
            ))
        }
        Opcode::Spawn3Indirect(op) => w.write_fmt(format_args!(
            "spawn3.indirect ${}, ${}:${}, ${} # {:?}",
            op.dest, op.module, op.function, op.args, op.opts
        )),
    }
}
