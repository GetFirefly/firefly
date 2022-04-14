use std::fmt;
use std::io::{self, Write};

use super::{Block, DataFlowGraph, Function, Inst, Value};

pub fn write_function(w: &mut dyn Write, func: &Function) -> io::Result<()> {
    let is_public = func.signature.visibility.is_public();
    if is_public {
        write!(w, "pub ")?;
    }
    write!(w, "function ")?;
    write_spec(w, func)?;
    if func.signature.visibility.is_externally_defined() {
        return Ok(());
    }
    writeln!(w, " {{")?;
    let mut any = false;
    for (block, block_data) in func.dfg.blocks() {
        if any {
            writeln!(w)?;
        }

        write_block_header(w, func, block, 4)?;
        for inst in block_data.insts() {
            write_instruction(w, func, inst, 4)?;
        }

        any = true;
    }
    writeln!(w, "}}")
}

fn write_spec(w: &mut dyn Write, func: &Function) -> io::Result<()> {
    write!(w, "{}(", &func.signature.name)?;
    let args = func
        .signature
        .params()
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<String>>()
        .join(", ");
    let results = func
        .signature
        .results()
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<String>>()
        .join(", ");
    write!(w, "{}) -> {} ", &args, &results)
}

fn write_arg(w: &mut dyn Write, func: &Function, arg: Value) -> io::Result<()> {
    write!(w, "{}: {}", arg, func.dfg.value_type(arg))
}

pub fn write_block_header(
    w: &mut dyn Write,
    func: &Function,
    block: Block,
    indent: usize,
) -> io::Result<()> {
    // The indent is for instructions, block header is 4 spaces outdented
    write!(w, "{1:0$}{2}", indent - 4, "", block)?;

    let mut args = func.dfg.block_params(block).iter().cloned();
    match args.next() {
        None => return writeln!(w, ":"),
        Some(arg) => {
            write!(w, "(")?;
            write_arg(w, func, arg)?;
        }
    }
    for arg in args {
        write!(w, ", ")?;
        write_arg(w, func, arg)?;
    }
    writeln!(w, "):")
}

fn write_instruction(
    w: &mut dyn Write,
    func: &Function,
    inst: Inst,
    indent: usize,
) -> io::Result<()> {
    let s = String::with_capacity(16);

    write!(w, "{1:0$}", indent, s)?;

    let mut has_results = false;
    for r in func.dfg.inst_results(inst) {
        if !has_results {
            has_results = true;
            write!(w, "{}", r)?;
        } else {
            write!(w, ", {}", r)?;
        }
    }
    if has_results {
        write!(w, " = ")?
    }

    let opcode = func.dfg[inst].opcode();
    write!(w, "{}", opcode)?;
    write_operands(w, &func.dfg, inst)?;

    if has_results {
        write!(w, "  : ")?;
        for (i, v) in func.dfg.inst_results(inst).iter().enumerate() {
            let t = func.dfg.value_type(*v).to_string();
            if i > 0 {
                write!(w, ", {}", t)?;
            } else {
                write!(w, "{}", t)?;
            }
        }
    }

    writeln!(w)?;

    Ok(())
}

fn write_operands(w: &mut dyn Write, dfg: &DataFlowGraph, inst: Inst) -> io::Result<()> {
    use crate::ir::*;
    use liblumen_binary::BinaryEntrySpecifier;

    let pool = &dfg.value_lists;
    match dfg[inst].as_ref() {
        InstData::BinaryOp(BinaryOp { args, .. }) => write!(w, " {}, {}", args[0], args[1]),
        InstData::BinaryOpImm(BinaryOpImm { arg, imm, .. }) => write!(w, " {}, {}", arg, imm),
        InstData::BinaryOpConst(BinaryOpConst { arg, imm, .. }) => {
            write!(w, " {}, ", arg)?;
            dfg.display_constant(w, *imm)
        }
        InstData::UnaryOp(UnaryOp { arg, .. }) => write!(w, " {}", arg),
        InstData::UnaryOpImm(UnaryOpImm { imm, .. }) => write!(w, " {}", imm),
        InstData::UnaryOpConst(UnaryOpConst { imm, .. }) => {
            write!(w, " ")?;
            dfg.display_constant(w, *imm)
        }
        InstData::Ret(Ret { args, .. }) => write!(w, " {}", DisplayValues(args.as_slice())),
        InstData::RetImm(RetImm { arg, imm, .. }) => write!(w, " {}, {}", arg, imm),
        InstData::Call(Call { args, .. }) => {
            let func_data = dfg.call_signature(inst).unwrap();
            write!(
                w,
                " {}({})",
                &func_data.mfa(),
                DisplayValues(args.as_slice(pool))
            )
        }
        InstData::CallIndirect(CallIndirect { callee, args, .. }) => {
            write!(w, " {:?}({})", &callee, DisplayValues(args.as_slice(pool)))
        }
        InstData::Br(Br {
            op,
            destination,
            args,
            ..
        }) if *op == Opcode::Br => {
            write!(w, " {}", destination)?;
            write_block_args(w, args.as_slice(pool))
        }
        InstData::Br(Br {
            destination, args, ..
        }) => {
            let args = args.as_slice(pool);
            write!(w, " {}, {}", args[0], destination)?;
            write_block_args(w, &args[1..])
        }
        InstData::PrimOp(PrimOp { args, .. }) => {
            write!(w, " {}", DisplayValues(args.as_slice(pool)))
        }
        InstData::PrimOpImm(PrimOpImm { imm, args, .. }) => {
            write!(w, " {}, {}", imm, DisplayValues(args.as_slice(pool)))
        }
        InstData::IsType(IsType { ty, arg, .. }) => {
            write!(w, " {}, {}", arg, ty)
        }
        InstData::BitsMatch(BitsMatch { spec, args, .. }) => {
            let values = DisplayValues(args.as_slice(pool));
            match spec {
                BinaryEntrySpecifier::Integer {
                    endianness, signed, ..
                } => {
                    if *signed {
                        write!(w, ".sint.{} {}", endianness, values)
                    } else {
                        write!(w, ".uint.{} {}", endianness, values)
                    }
                }
                BinaryEntrySpecifier::Float {
                    endianness, unit, ..
                } => {
                    write!(w, ".float.{}({}) {}", endianness, unit, values)
                }
                BinaryEntrySpecifier::Bits { unit, .. } => {
                    write!(w, ".bits({}) {}", unit, values)
                }
                BinaryEntrySpecifier::Bytes { unit, .. } => {
                    write!(w, ".bytes({}) {}", unit, values)
                }
                BinaryEntrySpecifier::Utf8 => {
                    write!(w, ".utf8 {}", values)
                }
                BinaryEntrySpecifier::Utf16 { endianness, .. } => {
                    write!(w, ".utf16.{} {}", endianness, values)
                }
                BinaryEntrySpecifier::Utf32 { endianness, .. } => {
                    write!(w, ".utf32.{} {}", endianness, values)
                }
            }
        }
        InstData::BitsPush(BitsPush { spec, args, .. }) => {
            let values = DisplayValues(args.as_slice(pool));
            match spec {
                BinaryEntrySpecifier::Integer {
                    endianness,
                    signed,
                    unit,
                    ..
                } => {
                    if *signed {
                        write!(w, ".sint.{}({}) {}", endianness, unit, values)
                    } else {
                        write!(w, ".uint.{}({}) {}", endianness, unit, values)
                    }
                }
                BinaryEntrySpecifier::Float {
                    endianness, unit, ..
                } => {
                    write!(w, ".float.{}({}) {}", endianness, unit, values)
                }
                BinaryEntrySpecifier::Bits { unit, .. } => {
                    write!(w, ".bits({}) {}", unit, values)
                }
                BinaryEntrySpecifier::Bytes { unit, .. } => {
                    write!(w, ".bytes({}) {}", unit, values)
                }
                BinaryEntrySpecifier::Utf8 => {
                    write!(w, ".utf8 {}", values)
                }
                BinaryEntrySpecifier::Utf16 { endianness, .. } => {
                    write!(w, ".utf16.{} {}", endianness, values)
                }
                BinaryEntrySpecifier::Utf32 { endianness, .. } => {
                    write!(w, ".utf32.{} {}", endianness, values)
                }
            }
        }
        InstData::SetElement(SetElement { args, .. }) => {
            let argv = args.as_slice();
            write!(w, " {}[{}], {}", argv[0], argv[1], argv[2])
        }
        InstData::SetElementImm(SetElementImm {
            arg, index, value, ..
        }) => write!(w, " {}[{}], {}", arg, index, value),
        InstData::SetElementConst(SetElementConst {
            arg, index, value, ..
        }) => write!(w, " {}[{}], {}", arg, index, value),
        InstData::MapUpdate(MapUpdate { args, .. }) => {
            write!(w, " {}[{}], {}", args[0], args[1], args[2])
        }
    }
}

fn write_block_args(w: &mut dyn Write, args: &[Value]) -> io::Result<()> {
    if args.is_empty() {
        Ok(())
    } else {
        write!(w, "({})", DisplayValues(args))
    }
}

struct DisplayValues<'a>(&'a [Value]);
impl<'a> fmt::Display for DisplayValues<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, val) in self.0.iter().enumerate() {
            if i == 0 {
                write!(f, "{}", val)?;
            } else {
                write!(f, ", {}", val)?;
            }
        }
        Ok(())
    }
}
