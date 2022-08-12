use liblumen_binary::{BinaryEntrySpecifier, BitVec};
use liblumen_diagnostics::SourceSpan;
use liblumen_intern::{Ident, Symbol};
use liblumen_number::{BigInt, Integer};
use liblumen_syntax_base::{PrimitiveType, TermType, Type};

use super::*;

pub trait InstBuilderBase<'f>: Sized {
    fn data_flow_graph(&self) -> &DataFlowGraph;
    fn data_flow_graph_mut(&mut self) -> &mut DataFlowGraph;
    fn build(self, data: InstData, ty: Type, span: SourceSpan) -> (Inst, &'f mut DataFlowGraph);
}

macro_rules! binary_op {
    ($name:ident, $op:expr) => {
        paste::paste! {
            fn $name(self, lhs: Value, rhs: Value, span: SourceSpan) -> Value {
                let ty = self.data_flow_graph().value_type(lhs);
                let (inst, dfg) = self.Binary($op, ty, lhs, rhs, span);
                dfg.first_result(inst)
            }
            fn [<$name _imm>](self, lhs: Value, rhs: Immediate, span: SourceSpan) -> Value {
                let ty = self.data_flow_graph().value_type(lhs);
                let (inst, dfg) = self.BinaryImm($op, ty, lhs, rhs, span);
                dfg.first_result(inst)
            }
            fn [<$name _const>](self, lhs: Value, rhs: Constant, span: SourceSpan) -> Value {
                let ty = self.data_flow_graph().value_type(lhs);
                let (inst, dfg) = self.BinaryConst($op, ty, lhs, rhs, span);
                dfg.first_result(inst)
            }
        }
    };
}
macro_rules! binary_bool_op {
    ($name:ident, $op:expr) => {
        paste::paste! {
            fn $name(self, lhs: Value, rhs: Value, span: SourceSpan) -> Value {
                let (inst, dfg) = self.Binary($op, Type::Term(TermType::Bool), lhs, rhs, span);
                dfg.first_result(inst)
            }
            fn [<$name _imm>](self, lhs: Value, rhs: bool, span: SourceSpan) -> Value {
                let (inst, dfg) = self.BinaryImm($op, Type::Term(TermType::Bool), lhs, Immediate::Bool(rhs), span);
                dfg.first_result(inst)
            }
        }
    };
}
macro_rules! binary_int_op {
    ($name:ident, $op:expr) => {
        paste::paste! {
            fn $name(self, lhs: Value, rhs: Value, span: SourceSpan) -> Value {
                let (inst, dfg) = self.Binary($op, Type::Term(TermType::Integer), lhs, rhs, span);
                dfg.first_result(inst)
            }
            fn [<$name _imm>](self, lhs: Value, rhs: i64, span: SourceSpan) -> Value {
                let (inst, dfg) = self.BinaryImm($op, Type::Term(TermType::Integer), lhs, Immediate::Integer(rhs), span);
                dfg.first_result(inst)
            }
            fn [<$name _const>](self, lhs: Value, rhs: Constant, span: SourceSpan) -> Value {
                let ty = self.data_flow_graph().constant_type(rhs);
                assert_eq!(ty, Type::Term(TermType::Integer), "invalid constant value type for integer op");
                let (inst, dfg) = self.BinaryConst($op, ty, lhs, rhs, span);
                dfg.first_result(inst)
            }
        }
    };
}
macro_rules! binary_numeric_op {
    ($name:ident, $op:expr) => {
        paste::paste! {
            fn $name(self, lhs: Value, rhs: Value, span: SourceSpan) -> Value {
                let lty = self.data_flow_graph().value_type(lhs);
                let rty = self.data_flow_graph().value_type(lhs).as_term().unwrap();
                let ty = lty.as_term().unwrap().coerce_to_numeric_with(rty);
                let (inst, dfg) = self.Binary($op, Type::Term(ty), lhs, rhs, span);
                dfg.first_result(inst)
            }
            fn [<$name _imm>](self, lhs: Value, rhs: Immediate, span: SourceSpan) -> Value {
                let lty = self.data_flow_graph().value_type(lhs);
                let rty = rhs.ty().as_term().unwrap();
                assert!(rty.is_numeric(), "invalid immediate value type for arithmetic op");
                let ty = lty.as_term().unwrap().coerce_to_numeric_with(rty);
                let (inst, dfg) = self.BinaryImm($op, Type::Term(ty), lhs, rhs, span);
                dfg.first_result(inst)
            }
            fn [<$name _const>](self, lhs: Value, rhs: Constant, span: SourceSpan) -> Value {
                let lty = self.data_flow_graph().value_type(lhs);
                let rty = self.data_flow_graph().constant_type(rhs).as_term().unwrap();
                assert!(rty.is_numeric(), "invalid constant value type for arithmetic op");
                let ty = lty.as_term().unwrap().coerce_to_numeric_with(rty);
                let (inst, dfg) = self.BinaryConst($op, Type::Term(ty), lhs, rhs, span);
                dfg.first_result(inst)
            }
        }
    };
}
macro_rules! unary_bool_op {
    ($name:ident, $op:expr) => {
        paste::paste! {
            fn $name(self, arg: Value, span: SourceSpan) -> Value {
                let (inst, dfg) = self.Unary($op, Type::Term(TermType::Bool), arg, span);
                dfg.first_result(inst)
            }
            fn [<$name _imm>](self, imm: bool, span: SourceSpan) -> Value {
                let (inst, dfg) = self.UnaryImm($op, Type::Term(TermType::Bool), Immediate::Bool(imm), span);
                dfg.first_result(inst)
            }
        }
    };
}
macro_rules! unary_int_op {
    ($name:ident, $op:expr) => {
        paste::paste! {
            fn $name(self, arg: Value, span: SourceSpan) -> Value {
                let (inst, dfg) = self.Unary($op, Type::Term(TermType::Integer), arg, span);
                dfg.first_result(inst)
            }
            fn [<$name _imm>](self, imm: i64, span: SourceSpan) -> Value {
                let (inst, dfg) = self.UnaryImm($op, Type::Term(TermType::Integer), Immediate::Integer(imm), span);
                dfg.first_result(inst)
            }
            fn [<$name _const>](self, imm: Constant, span: SourceSpan) -> Value {
                let ty = self.data_flow_graph().constant_type(imm);
                assert_eq!(ty, Type::Term(TermType::Integer), "invalid constant value for integer op");
                let (inst, dfg) = self.UnaryConst($op, ty, imm, span);
                dfg.first_result(inst)
            }
        }
    };
}
macro_rules! unary_numeric_op {
    ($name:ident, $op:expr) => {
        paste::paste! {
            fn $name(self, arg: Value, span: SourceSpan) -> Value {
                let ty = self.data_flow_graph().value_type(arg);
                let (inst, dfg) = self.Unary($op, Type::Term(ty.as_term().unwrap().coerce_to_numeric()), arg, span);
                dfg.first_result(inst)
            }
            fn [<$name _imm>](self, imm: Immediate, span: SourceSpan) -> Value {
                let ty = imm.ty().as_term().unwrap();
                assert!(ty.is_numeric(), "invalid immediate value for arithmetic op");
                let (inst, dfg) = self.UnaryImm($op, Type::Term(ty), imm, span);
                dfg.first_result(inst)
            }
            fn [<$name _const>](self, imm: Constant, span: SourceSpan) -> Value {
                let ty = self.data_flow_graph().constant_type(imm);
                assert!(ty.as_term().unwrap().is_numeric(), "invalid constant value for arithmetic op");
                let (inst, dfg) = self.UnaryConst($op, ty, imm, span);
                dfg.first_result(inst)
            }
        }
    };
}

pub trait InstBuilder<'f>: InstBuilderBase<'f> {
    fn int(self, i: i64, span: SourceSpan) -> Value {
        let (inst, dfg) = self.UnaryImm(
            Opcode::ImmInt,
            Type::Term(TermType::Integer),
            Immediate::Integer(i),
            span,
        );
        dfg.first_result(inst)
    }

    fn bigint(mut self, i: BigInt, span: SourceSpan) -> Value {
        let constant = {
            self.data_flow_graph_mut()
                .make_constant(ConstantItem::Integer(Integer::Big(i)))
        };
        let (inst, dfg) = self.UnaryConst(
            Opcode::ConstBigInt,
            Type::Term(TermType::Integer),
            constant,
            span,
        );
        dfg.first_result(inst)
    }

    fn character(self, c: char, span: SourceSpan) -> Value {
        let (inst, dfg) = self.UnaryImm(
            Opcode::ImmInt,
            Type::Term(TermType::Integer),
            Immediate::Integer((c as u32).try_into().unwrap()),
            span,
        );
        dfg.first_result(inst)
    }

    fn float(self, f: f64, span: SourceSpan) -> Value {
        let (inst, dfg) = self.UnaryImm(
            Opcode::ImmFloat,
            Type::Term(TermType::Float),
            Immediate::Float(f),
            span,
        );
        dfg.first_result(inst)
    }

    fn bool(self, b: bool, span: SourceSpan) -> Value {
        let (inst, dfg) = self.UnaryImm(
            Opcode::ImmBool,
            Type::Term(TermType::Bool),
            Immediate::Bool(b),
            span,
        );
        dfg.first_result(inst)
    }

    fn atom(self, a: Symbol, span: SourceSpan) -> Value {
        let (inst, dfg) = self.UnaryImm(
            Opcode::ImmAtom,
            Type::Term(TermType::Atom),
            Immediate::Atom(a),
            span,
        );
        dfg.first_result(inst)
    }

    fn nil(self, span: SourceSpan) -> Value {
        let (inst, dfg) = self.UnaryImm(
            Opcode::ImmNil,
            Type::Term(TermType::Nil),
            Immediate::Nil,
            span,
        );
        dfg.first_result(inst)
    }

    fn none(self, span: SourceSpan) -> Value {
        let (inst, dfg) = self.UnaryImm(
            Opcode::ImmNone,
            Type::Term(TermType::Any),
            Immediate::None,
            span,
        );
        dfg.first_result(inst)
    }

    fn null(self, ty: Type, span: SourceSpan) -> Value {
        let (inst, dfg) = self.UnaryImm(Opcode::ImmNull, ty, Immediate::None, span);
        dfg.first_result(inst)
    }

    fn cast(self, arg: Value, ty: Type, span: SourceSpan) -> Value {
        let (inst, dfg) = self.Unary(Opcode::Cast, ty, arg, span);
        dfg.first_result(inst)
    }

    fn binary_from_ident(mut self, id: Ident) -> Value {
        let constant = {
            self.data_flow_graph_mut()
                .make_constant(ConstantItem::InternedStr(id.name))
        };
        let (inst, dfg) = self.UnaryConst(
            Opcode::ConstBinary,
            Type::Term(TermType::Binary),
            constant,
            id.span,
        );
        dfg.first_result(inst)
    }

    fn binary_from_string(mut self, s: String, span: SourceSpan) -> Value {
        let constant = {
            self.data_flow_graph_mut()
                .make_constant(ConstantItem::String(s))
        };
        let (inst, dfg) = self.UnaryConst(
            Opcode::ConstBinary,
            Type::Term(TermType::Binary),
            constant,
            span,
        );
        dfg.first_result(inst)
    }

    fn binary_from_bytes(mut self, bytes: &[u8], span: SourceSpan) -> Value {
        let constant = {
            self.data_flow_graph_mut()
                .make_constant(ConstantItem::Bytes(bytes.to_vec().into()))
        };
        let (inst, dfg) = self.UnaryConst(
            Opcode::ConstBinary,
            Type::Term(TermType::Binary),
            constant,
            span,
        );
        dfg.first_result(inst)
    }

    fn bitstring(mut self, bitvec: BitVec, span: SourceSpan) -> Value {
        let constant = {
            self.data_flow_graph_mut()
                .make_constant(ConstantItem::Bitstring(bitvec))
        };
        let (inst, dfg) = self.UnaryConst(
            Opcode::ConstBinary,
            Type::Term(TermType::Binary),
            constant,
            span,
        );
        dfg.first_result(inst)
    }

    fn tuple_const(mut self, elements: &[Constant], span: SourceSpan) -> Value {
        let constant = {
            self.data_flow_graph_mut()
                .make_constant(ConstantItem::Tuple(elements.to_vec()))
        };
        let (inst, dfg) = self.UnaryConst(
            Opcode::ConstTuple,
            Type::Term(TermType::Tuple(None)),
            constant,
            span,
        );
        dfg.first_result(inst)
    }

    fn list_const(mut self, elements: &[Constant], span: SourceSpan) -> Value {
        let constant = {
            self.data_flow_graph_mut()
                .make_constant(ConstantItem::List(elements.to_vec()))
        };
        let (inst, dfg) = self.UnaryConst(
            Opcode::ConstList,
            Type::Term(TermType::List(None)),
            constant,
            span,
        );
        dfg.first_result(inst)
    }

    fn map_const(mut self, pairs: &[(Constant, Constant)], span: SourceSpan) -> Value {
        let constant = {
            self.data_flow_graph_mut()
                .make_constant(ConstantItem::Map(pairs.to_vec()))
        };
        let (inst, dfg) =
            self.UnaryConst(Opcode::ConstMap, Type::Term(TermType::Map), constant, span);
        dfg.first_result(inst)
    }

    fn is_null(self, arg: Value, span: SourceSpan) -> Value {
        let (inst, dfg) = self.Unary(
            Opcode::IsNull,
            Type::Primitive(PrimitiveType::I1),
            arg,
            span,
        );
        dfg.first_result(inst)
    }

    fn trunc(self, arg: Value, ty: PrimitiveType, span: SourceSpan) -> Value {
        let (inst, dfg) = self.Unary(Opcode::Trunc, Type::Primitive(ty), arg, span);
        dfg.first_result(inst)
    }

    fn zext(self, arg: Value, ty: PrimitiveType, span: SourceSpan) -> Value {
        let (inst, dfg) = self.Unary(Opcode::Zext, Type::Primitive(ty), arg, span);
        dfg.first_result(inst)
    }

    fn zext_imm(self, imm: i64, ty: PrimitiveType, span: SourceSpan) -> Value {
        let (inst, dfg) = self.UnaryImm(
            Opcode::Zext,
            Type::Primitive(ty),
            Immediate::Integer(imm),
            span,
        );
        dfg.first_result(inst)
    }

    fn icmp_eq(self, lhs: Value, rhs: Value, span: SourceSpan) -> Value {
        let (inst, dfg) = self.Binary(
            Opcode::IcmpEq,
            Type::Primitive(PrimitiveType::I1),
            lhs,
            rhs,
            span,
        );
        dfg.first_result(inst)
    }

    fn icmp_eq_imm(self, lhs: Value, rhs: i64, span: SourceSpan) -> Value {
        let (inst, dfg) = self.BinaryImm(
            Opcode::IcmpEq,
            Type::Primitive(PrimitiveType::I1),
            lhs,
            Immediate::Integer(rhs),
            span,
        );
        dfg.first_result(inst)
    }

    fn icmp_neq(self, lhs: Value, rhs: Value, span: SourceSpan) -> Value {
        let (inst, dfg) = self.Binary(
            Opcode::IcmpNeq,
            Type::Primitive(PrimitiveType::I1),
            lhs,
            rhs,
            span,
        );
        dfg.first_result(inst)
    }

    fn icmp_neq_imm(self, lhs: Value, rhs: i64, span: SourceSpan) -> Value {
        let (inst, dfg) = self.BinaryImm(
            Opcode::IcmpNeq,
            Type::Primitive(PrimitiveType::I1),
            lhs,
            Immediate::Integer(rhs),
            span,
        );
        dfg.first_result(inst)
    }

    fn icmp_gt(self, lhs: Value, rhs: Value, span: SourceSpan) -> Value {
        let (inst, dfg) = self.Binary(
            Opcode::IcmpGt,
            Type::Primitive(PrimitiveType::I1),
            lhs,
            rhs,
            span,
        );
        dfg.first_result(inst)
    }

    fn icmp_gt_imm(self, lhs: Value, rhs: i64, span: SourceSpan) -> Value {
        let (inst, dfg) = self.BinaryImm(
            Opcode::IcmpGt,
            Type::Primitive(PrimitiveType::I1),
            lhs,
            Immediate::Integer(rhs),
            span,
        );
        dfg.first_result(inst)
    }

    fn icmp_gte(self, lhs: Value, rhs: Value, span: SourceSpan) -> Value {
        let (inst, dfg) = self.Binary(
            Opcode::IcmpGte,
            Type::Primitive(PrimitiveType::I1),
            lhs,
            rhs,
            span,
        );
        dfg.first_result(inst)
    }

    fn icmp_gte_imm(self, lhs: Value, rhs: i64, span: SourceSpan) -> Value {
        let (inst, dfg) = self.BinaryImm(
            Opcode::IcmpGte,
            Type::Primitive(PrimitiveType::I1),
            lhs,
            Immediate::Integer(rhs),
            span,
        );
        dfg.first_result(inst)
    }

    fn icmp_lt(self, lhs: Value, rhs: Value, span: SourceSpan) -> Value {
        let (inst, dfg) = self.Binary(
            Opcode::IcmpLt,
            Type::Primitive(PrimitiveType::I1),
            lhs,
            rhs,
            span,
        );
        dfg.first_result(inst)
    }

    fn icmp_lt_imm(self, lhs: Value, rhs: i64, span: SourceSpan) -> Value {
        let (inst, dfg) = self.BinaryImm(
            Opcode::IcmpLt,
            Type::Primitive(PrimitiveType::I1),
            lhs,
            Immediate::Integer(rhs),
            span,
        );
        dfg.first_result(inst)
    }

    fn icmp_lte(self, lhs: Value, rhs: Value, span: SourceSpan) -> Value {
        let (inst, dfg) = self.Binary(
            Opcode::IcmpLte,
            Type::Primitive(PrimitiveType::I1),
            lhs,
            rhs,
            span,
        );
        dfg.first_result(inst)
    }

    fn icmp_lte_imm(self, lhs: Value, rhs: i64, span: SourceSpan) -> Value {
        let (inst, dfg) = self.BinaryImm(
            Opcode::IcmpLte,
            Type::Primitive(PrimitiveType::I1),
            lhs,
            Immediate::Integer(rhs),
            span,
        );
        dfg.first_result(inst)
    }

    fn make_fun(mut self, callee: FuncRef, env: &[Value], span: SourceSpan) -> Inst {
        let mut vlist = ValueList::default();
        {
            let pool = &mut self.data_flow_graph_mut().value_lists;
            vlist.extend(env.iter().copied(), pool);
        }
        self.MakeFun(Type::Term(TermType::Fun(None)), callee, vlist, span)
            .0
    }

    fn br(mut self, block: Block, args: &[Value], span: SourceSpan) -> Inst {
        let mut vlist = ValueList::default();
        {
            let pool = &mut self.data_flow_graph_mut().value_lists;
            vlist.extend(args.iter().copied(), pool);
        }
        self.Br(Opcode::Br, Type::Invalid, block, vlist, span).0
    }

    fn br_if(mut self, cond: Value, block: Block, args: &[Value], span: SourceSpan) -> Inst {
        let ty = self.data_flow_graph().value_type(cond);
        let mut vlist = ValueList::default();
        {
            let pool = &mut self.data_flow_graph_mut().value_lists;
            vlist.push(cond, pool);
            vlist.extend(args.iter().copied(), pool);
        }
        self.Br(Opcode::BrIf, ty, block, vlist, span).0
    }

    fn br_unless(mut self, cond: Value, block: Block, args: &[Value], span: SourceSpan) -> Inst {
        let ty = self.data_flow_graph().value_type(cond);
        let mut vlist = ValueList::default();
        {
            let pool = &mut self.data_flow_graph_mut().value_lists;
            vlist.push(cond, pool);
            vlist.extend(args.iter().copied(), pool);
        }
        self.Br(Opcode::BrUnless, ty, block, vlist, span).0
    }

    fn cond_br(
        mut self,
        cond: Value,
        then_dest: Block,
        then_args: &[Value],
        else_dest: Block,
        else_args: &[Value],
        span: SourceSpan,
    ) -> Inst {
        let mut then_vlist = ValueList::default();
        let mut else_vlist = ValueList::default();
        {
            let pool = &mut self.data_flow_graph_mut().value_lists;
            then_vlist.extend(then_args.iter().copied(), pool);
            else_vlist.extend(else_args.iter().copied(), pool);
        }
        self.CondBr(cond, then_dest, then_vlist, else_dest, else_vlist, span)
            .0
    }

    fn switch(self, arg: Value, arms: Vec<(u32, Block)>, default: Block, span: SourceSpan) -> Inst {
        self.Switch(arg, arms, default, span).0
    }

    fn ret(self, is_err: Value, returning: Value, span: SourceSpan) -> Inst {
        self.Ret(is_err, returning, span).0
    }

    fn ret_ok(self, returning: Value, span: SourceSpan) -> Inst {
        self.RetImm(Immediate::Bool(false), returning, span).0
    }

    fn ret_err(self, returning: Value, span: SourceSpan) -> Inst {
        self.RetImm(Immediate::Bool(true), returning, span).0
    }

    fn call(mut self, callee: FuncRef, args: &[Value], span: SourceSpan) -> Inst {
        let mut vlist = ValueList::default();
        {
            let pool = &mut self.data_flow_graph_mut().value_lists;
            vlist.extend(args.iter().copied(), pool);
        }
        self.Call(callee, vlist, false, span).0
    }

    fn call_indirect(mut self, callee: Value, args: &[Value], span: SourceSpan) -> Inst {
        let mut vlist = ValueList::default();
        {
            let pool = &mut self.data_flow_graph_mut().value_lists;
            vlist.extend(args.iter().copied(), pool);
        }
        self.CallIndirect(callee, vlist, false, span).0
    }

    fn enter(mut self, callee: FuncRef, args: &[Value], span: SourceSpan) -> Inst {
        let mut vlist = ValueList::default();
        {
            let pool = &mut self.data_flow_graph_mut().value_lists;
            vlist.extend(args.iter().copied(), pool);
        }
        self.Call(callee, vlist, true, span).0
    }

    fn enter_indirect(mut self, callee: Value, args: &[Value], span: SourceSpan) -> Inst {
        let mut vlist = ValueList::default();
        {
            let pool = &mut self.data_flow_graph_mut().value_lists;
            vlist.extend(args.iter().copied(), pool);
        }
        self.CallIndirect(callee, vlist, true, span).0
    }

    fn is_type(self, ty: Type, value: Value, span: SourceSpan) -> Value {
        let (inst, dfg) = self.IsType(ty, value, span);
        dfg.first_result(inst)
    }

    fn cons(self, head: Value, tail: Value, span: SourceSpan) -> Value {
        let (inst, dfg) = self.Binary(
            Opcode::Cons,
            Type::Term(TermType::List(None)),
            head,
            tail,
            span,
        );
        dfg.first_result(inst)
    }

    fn cons_imm(self, head: Value, tail: Immediate, span: SourceSpan) -> Value {
        let (inst, dfg) = self.BinaryImm(
            Opcode::Cons,
            Type::Term(TermType::List(None)),
            head,
            tail,
            span,
        );
        dfg.first_result(inst)
    }

    fn cons_const(self, head: Value, tail: Constant, span: SourceSpan) -> Value {
        let (inst, dfg) = self.BinaryConst(
            Opcode::Cons,
            Type::Term(TermType::List(None)),
            head,
            tail,
            span,
        );
        dfg.first_result(inst)
    }

    fn head(self, list: Value, span: SourceSpan) -> Value {
        let (inst, dfg) = self.Unary(Opcode::Head, Type::Term(TermType::Any), list, span);
        dfg.first_result(inst)
    }

    fn tail(self, list: Value, span: SourceSpan) -> Value {
        let (inst, dfg) = self.Unary(
            Opcode::Tail,
            Type::Term(TermType::MaybeImproperList),
            list,
            span,
        );
        dfg.first_result(inst)
    }

    fn tuple_imm(self, arity: usize, span: SourceSpan) -> Value {
        let (inst, dfg) = self.UnaryImm(
            Opcode::Tuple,
            Type::Term(TermType::Tuple(None)),
            Immediate::Integer(arity.try_into().unwrap()),
            span,
        );
        dfg.first_result(inst)
    }

    fn map(self, span: SourceSpan) -> Value {
        let vlist = ValueList::default();
        let (inst, dfg) = self.PrimOp(Opcode::Map, Type::Term(TermType::Map), vlist, span);
        dfg.first_result(inst)
    }

    fn map_get(mut self, map: Value, key: Value, span: SourceSpan) -> Value {
        let mut vlist = ValueList::default();
        {
            let pool = &mut self.data_flow_graph_mut().value_lists;
            vlist.push(map, pool);
            vlist.push(key, pool);
        }
        let (inst, dfg) = self.PrimOp(Opcode::MapGet, Type::Term(TermType::Any), vlist, span);
        dfg.first_result(inst)
    }

    fn map_fetch(mut self, map: Value, key: Value, span: SourceSpan) -> Inst {
        let mut vlist = ValueList::default();
        {
            let pool = &mut self.data_flow_graph_mut().value_lists;
            vlist.push(map, pool);
            vlist.push(key, pool);
        }
        self.PrimOp(Opcode::MapFetch, Type::Term(TermType::Any), vlist, span)
            .0
    }

    fn map_put(self, map: Value, key: Value, value: Value, span: SourceSpan) -> Value {
        let (inst, dfg) = self.MapUpdate(Opcode::MapPut, map, key, value, span);
        dfg.first_result(inst)
    }

    fn map_put_mut(self, map: Value, key: Value, value: Value, span: SourceSpan) -> Value {
        let (inst, dfg) = self.MapUpdate(Opcode::MapPutMut, map, key, value, span);
        dfg.first_result(inst)
    }

    fn map_update(self, map: Value, key: Value, value: Value, span: SourceSpan) -> Inst {
        self.MapUpdate(Opcode::MapUpdate, map, key, value, span).0
    }

    fn map_update_mut(self, map: Value, key: Value, value: Value, span: SourceSpan) -> Inst {
        self.MapUpdate(Opcode::MapUpdateMut, map, key, value, span)
            .0
    }

    fn get_element(self, tuple: Value, index: Value, span: SourceSpan) -> Value {
        let (inst, dfg) = self.Binary(
            Opcode::GetElement,
            Type::Term(TermType::Any),
            tuple,
            index,
            span,
        );
        dfg.first_result(inst)
    }

    fn get_element_imm(self, tuple: Value, index: Immediate, span: SourceSpan) -> Value {
        let (inst, dfg) = self.BinaryImm(
            Opcode::GetElement,
            Type::Term(TermType::Any),
            tuple,
            index,
            span,
        );
        dfg.first_result(inst)
    }

    fn set_element(self, tuple: Value, index: usize, value: Value, span: SourceSpan) -> Value {
        let index = Immediate::Integer(index.try_into().unwrap());
        let (inst, dfg) = self.SetElement(Opcode::SetElement, tuple, index, value, span);
        dfg.first_result(inst)
    }

    fn set_element_mut(self, tuple: Value, index: usize, value: Value, span: SourceSpan) -> Value {
        let index = Immediate::Integer(index.try_into().unwrap());
        let (inst, dfg) = self.SetElement(Opcode::SetElementMut, tuple, index, value, span);
        dfg.first_result(inst)
    }

    fn set_element_imm(
        self,
        tuple: Value,
        index: usize,
        value: Immediate,
        span: SourceSpan,
    ) -> Value {
        let index = Immediate::Integer(index.try_into().unwrap());
        let (inst, dfg) = self.SetElementImm(Opcode::SetElement, tuple, index, value, span);
        dfg.first_result(inst)
    }

    fn set_element_mut_imm(
        self,
        tuple: Value,
        index: usize,
        value: Immediate,
        span: SourceSpan,
    ) -> Value {
        let index = Immediate::Integer(index.try_into().unwrap());
        let (inst, dfg) = self.SetElementImm(Opcode::SetElementMut, tuple, index, value, span);
        dfg.first_result(inst)
    }

    fn set_element_const(
        self,
        tuple: Value,
        index: usize,
        value: Constant,
        span: SourceSpan,
    ) -> Value {
        let index = Immediate::Integer(index.try_into().unwrap());
        let (inst, dfg) = self.SetElementConst(tuple, index, value, span);
        dfg.first_result(inst)
    }

    fn match_fail(mut self, value: Value, span: SourceSpan) -> Inst {
        let mut vlist = ValueList::default();
        {
            let pool = &mut self.data_flow_graph_mut().value_lists;
            vlist.push(value, pool);
        }
        self.PrimOp(Opcode::MatchFail, Type::Invalid, vlist, span).0
    }

    fn match_fail_imm(self, reason: Immediate, span: SourceSpan) -> Inst {
        let vlist = ValueList::default();
        self.PrimOpImm(Opcode::MatchFail, Type::Invalid, reason, vlist, span)
            .0
    }

    fn recv_start(mut self, timeout: Value, span: SourceSpan) -> Value {
        let mut vlist = ValueList::default();
        {
            let pool = &mut self.data_flow_graph_mut().value_lists;
            vlist.push(timeout, pool);
        }
        let (inst, dfg) = self.PrimOp(Opcode::RecvStart, Type::RecvContext, vlist, span);
        dfg.first_result(inst)
    }

    fn recv_next(mut self, recv_context: Value, span: SourceSpan) -> Value {
        let mut vlist = ValueList::default();
        {
            let pool = &mut self.data_flow_graph_mut().value_lists;
            vlist.push(recv_context, pool);
        }
        let (inst, dfg) = self.PrimOp(Opcode::RecvNext, Type::RecvState, vlist, span);
        dfg.first_result(inst)
    }

    fn recv_peek(mut self, recv_context: Value, span: SourceSpan) -> Value {
        let mut vlist = ValueList::default();
        {
            let pool = &mut self.data_flow_graph_mut().value_lists;
            vlist.push(recv_context, pool);
        }
        let (inst, dfg) = self.PrimOp(Opcode::RecvPeek, Type::Term(TermType::Any), vlist, span);
        dfg.first_result(inst)
    }

    fn recv_pop(mut self, recv_context: Value, span: SourceSpan) -> Inst {
        let mut vlist = ValueList::default();
        {
            let pool = &mut self.data_flow_graph_mut().value_lists;
            vlist.push(recv_context, pool);
        }
        self.PrimOp(Opcode::RecvPop, Type::Invalid, vlist, span).0
    }

    fn recv_wait(mut self, recv_context: Value, span: SourceSpan) -> Inst {
        let mut vlist = ValueList::default();
        {
            let pool = &mut self.data_flow_graph_mut().value_lists;
            vlist.push(recv_context, pool);
        }
        self.PrimOp(Opcode::RecvWait, Type::Invalid, vlist, span).0
    }

    fn bs_init_writable(self, span: SourceSpan) -> Inst {
        let vlist = ValueList::default();
        self.PrimOp(Opcode::BitsInitWritable, Type::BinaryBuilder, vlist, span)
            .0
    }

    fn bs_init_writable_imm(self, size: usize, span: SourceSpan) -> Inst {
        let vlist = ValueList::default();
        self.PrimOpImm(
            Opcode::BitsInitWritable,
            Type::BinaryBuilder,
            Immediate::Integer(size.try_into().unwrap()),
            vlist,
            span,
        )
        .0
    }

    fn bs_close_writable(mut self, bin: Value, span: SourceSpan) -> Inst {
        let mut vlist = ValueList::default();
        {
            let pool = &mut self.data_flow_graph_mut().value_lists;
            vlist.push(bin, pool);
        }
        self.PrimOp(
            Opcode::BitsCloseWritable,
            Type::Term(TermType::Bitstring),
            vlist,
            span,
        )
        .0
    }

    fn bs_test_tail_imm(self, bin: Value, imm: Immediate, span: SourceSpan) -> Value {
        let (inst, dfg) = self.BinaryImm(
            Opcode::BitsTestTail,
            Type::Primitive(PrimitiveType::I1),
            bin,
            imm,
            span,
        );
        dfg.first_result(inst)
    }

    fn bs_start_match(mut self, bin: Value, span: SourceSpan) -> Inst {
        let mut vlist = ValueList::default();
        {
            let pool = &mut self.data_flow_graph_mut().value_lists;
            vlist.push(bin, pool);
        }
        self.PrimOp(Opcode::BitsStartMatch, Type::MatchContext, vlist, span)
            .0
    }

    fn bs_match(
        mut self,
        spec: BinaryEntrySpecifier,
        bin: Value,
        size: Option<Value>,
        span: SourceSpan,
    ) -> Inst {
        let mut vlist = ValueList::default();
        {
            let pool = &mut self.data_flow_graph_mut().value_lists;
            vlist.push(bin, pool);
            if let Some(sz) = size {
                vlist.push(sz, pool);
            }
        }
        self.BitsMatch(spec, vlist, span).0
    }

    fn bs_push(
        mut self,
        spec: BinaryEntrySpecifier,
        bin: Value,
        value: Value,
        size: Option<Value>,
        span: SourceSpan,
    ) -> Inst {
        let mut vlist = ValueList::default();
        {
            let pool = &mut self.data_flow_graph_mut().value_lists;
            vlist.push(bin, pool);
            vlist.push(value, pool);
            if let Some(sz) = size {
                vlist.push(sz, pool);
            }
        }
        self.BitsPush(spec, vlist, span).0
    }

    fn raise(mut self, class: Value, error: Value, trace: Value, span: SourceSpan) -> Inst {
        let mut vlist = ValueList::default();
        {
            let pool = &mut self.data_flow_graph_mut().value_lists;
            vlist.push(class, pool);
            vlist.push(error, pool);
            vlist.push(trace, pool);
        }
        self.PrimOp(Opcode::Raise, Type::Invalid, vlist, span).0
    }

    fn build_stacktrace(self, span: SourceSpan) -> Value {
        let vlist = ValueList::default();
        let (inst, dfg) = self.PrimOp(
            Opcode::BuildStacktrace,
            Type::Term(TermType::Any),
            vlist,
            span,
        );
        dfg.first_result(inst)
    }

    fn exception_class(mut self, exception: Value, span: SourceSpan) -> Value {
        let mut vlist = ValueList::default();
        {
            let pool = &mut self.data_flow_graph_mut().value_lists;
            vlist.push(exception, pool);
        }
        let (inst, dfg) = self.PrimOp(
            Opcode::ExceptionClass,
            Type::Term(TermType::Atom),
            vlist,
            span,
        );
        dfg.first_result(inst)
    }

    fn exception_reason(mut self, exception: Value, span: SourceSpan) -> Value {
        let mut vlist = ValueList::default();
        {
            let pool = &mut self.data_flow_graph_mut().value_lists;
            vlist.push(exception, pool);
        }
        let (inst, dfg) = self.PrimOp(
            Opcode::ExceptionReason,
            Type::Term(TermType::Any),
            vlist,
            span,
        );
        dfg.first_result(inst)
    }

    fn exception_trace(mut self, exception: Value, span: SourceSpan) -> Value {
        let mut vlist = ValueList::default();
        {
            let pool = &mut self.data_flow_graph_mut().value_lists;
            vlist.push(exception, pool);
        }
        let (inst, dfg) = self.PrimOp(
            Opcode::ExceptionTrace,
            Type::Term(TermType::List(None)),
            vlist,
            span,
        );
        dfg.first_result(inst)
    }

    binary_op!(is_tagged_tuple, Opcode::IsTaggedTuple);
    binary_op!(eq, Opcode::Eq);
    binary_op!(eq_exact, Opcode::EqExact);
    binary_op!(neq, Opcode::Neq);
    binary_op!(neq_exact, Opcode::NeqExact);
    binary_op!(gt, Opcode::Gt);
    binary_op!(gte, Opcode::Gte);
    binary_op!(lt, Opcode::Lt);
    binary_op!(lte, Opcode::Lte);
    binary_bool_op!(and, Opcode::And);
    binary_bool_op!(andalso, Opcode::AndAlso);
    binary_bool_op!(or, Opcode::Or);
    binary_bool_op!(orelse, Opcode::OrElse);
    binary_bool_op!(xor, Opcode::Xor);
    binary_int_op!(band, Opcode::Band);
    binary_int_op!(bor, Opcode::Bor);
    binary_int_op!(bxor, Opcode::Bxor);
    binary_int_op!(div, Opcode::Div);
    binary_int_op!(rem, Opcode::Rem);
    binary_int_op!(bsl, Opcode::Bsl);
    binary_int_op!(bsr, Opcode::Bsr);
    binary_numeric_op!(add, Opcode::Add);
    binary_numeric_op!(sub, Opcode::Sub);
    binary_numeric_op!(mul, Opcode::Mul);
    binary_numeric_op!(fdiv, Opcode::Fdiv);
    unary_numeric_op!(neg, Opcode::Neg);
    unary_bool_op!(not, Opcode::Not);
    unary_int_op!(bnot, Opcode::Bnot);

    fn list_concat(self, lhs: Value, rhs: Value, span: SourceSpan) -> Value {
        let (inst, dfg) = self.Binary(
            Opcode::ListConcat,
            Type::Term(TermType::List(None)),
            lhs,
            rhs,
            span,
        );
        dfg.first_result(inst)
    }

    fn list_concat_const(self, lhs: Value, rhs: Constant, span: SourceSpan) -> Value {
        let ty = self.data_flow_graph().constant_type(rhs);
        assert!(
            ty.as_term().unwrap().is_list(),
            "invalid constant value type for list concatenation"
        );
        let (inst, dfg) = self.BinaryConst(
            Opcode::ListConcat,
            Type::Term(TermType::List(None)),
            lhs,
            rhs,
            span,
        );
        dfg.first_result(inst)
    }

    fn list_subtract(self, lhs: Value, rhs: Value, span: SourceSpan) -> Value {
        let (inst, dfg) = self.Binary(
            Opcode::ListSubtract,
            Type::Term(TermType::List(None)),
            lhs,
            rhs,
            span,
        );
        dfg.first_result(inst)
    }

    fn list_subtract_const(self, lhs: Value, rhs: Constant, span: SourceSpan) -> Value {
        let ty = self.data_flow_graph().constant_type(rhs);
        assert!(
            ty.as_term().unwrap().is_list(),
            "invalid constant value type for list subtraction"
        );
        let (inst, dfg) = self.BinaryConst(
            Opcode::ListSubtract,
            Type::Term(TermType::List(None)),
            lhs,
            rhs,
            span,
        );
        dfg.first_result(inst)
    }

    #[allow(non_snake_case)]
    fn MakeFun(
        self,
        ty: Type,
        callee: FuncRef,
        env: ValueList,
        span: SourceSpan,
    ) -> (Inst, &'f mut DataFlowGraph) {
        let data = InstData::MakeFun(MakeFun { callee, env });
        self.build(data, ty, span)
    }

    #[allow(non_snake_case)]
    fn CondBr(
        self,
        cond: Value,
        then_dest: Block,
        then_args: ValueList,
        else_dest: Block,
        else_args: ValueList,
        span: SourceSpan,
    ) -> (Inst, &'f mut DataFlowGraph) {
        let data = InstData::CondBr(CondBr {
            cond,
            then_dest: (then_dest, then_args),
            else_dest: (else_dest, else_args),
        });
        self.build(data, Type::Invalid, span)
    }

    #[allow(non_snake_case)]
    fn Br(
        self,
        op: Opcode,
        ty: Type,
        destination: Block,
        args: ValueList,
        span: SourceSpan,
    ) -> (Inst, &'f mut DataFlowGraph) {
        let data = InstData::Br(Br {
            op,
            destination,
            args,
        });
        self.build(data, ty, span)
    }

    #[allow(non_snake_case)]
    fn Switch(
        self,
        arg: Value,
        arms: Vec<(u32, Block)>,
        default: Block,
        span: SourceSpan,
    ) -> (Inst, &'f mut DataFlowGraph) {
        let data = InstData::Switch(Switch {
            op: Opcode::Switch,
            arg,
            arms,
            default,
        });
        self.build(data, Type::Invalid, span)
    }

    #[allow(non_snake_case)]
    fn Ret(
        self,
        is_err: Value,
        returning: Value,
        span: SourceSpan,
    ) -> (Inst, &'f mut DataFlowGraph) {
        let data = InstData::Ret(Ret {
            op: Opcode::Ret,
            args: [is_err, returning],
        });
        self.build(data, Type::Invalid, span)
    }

    #[allow(non_snake_case)]
    fn RetImm(
        self,
        is_err: Immediate,
        returning: Value,
        span: SourceSpan,
    ) -> (Inst, &'f mut DataFlowGraph) {
        let data = InstData::RetImm(RetImm {
            op: Opcode::Ret,
            imm: is_err,
            arg: returning,
        });
        self.build(data, Type::Invalid, span)
    }

    #[allow(non_snake_case)]
    fn Call(
        self,
        callee: FuncRef,
        args: ValueList,
        is_tail: bool,
        span: SourceSpan,
    ) -> (Inst, &'f mut DataFlowGraph) {
        let data = InstData::Call(Call {
            callee,
            args,
            is_tail,
        });
        self.build(data, Type::Invalid, span)
    }

    #[allow(non_snake_case)]
    fn CallIndirect(
        self,
        callee: Value,
        args: ValueList,
        is_tail: bool,
        span: SourceSpan,
    ) -> (Inst, &'f mut DataFlowGraph) {
        let data = InstData::CallIndirect(CallIndirect {
            callee,
            args,
            is_tail,
        });
        self.build(data, Type::Invalid, span)
    }

    #[allow(non_snake_case)]
    fn IsType(self, ty: Type, arg: Value, span: SourceSpan) -> (Inst, &'f mut DataFlowGraph) {
        let data = InstData::IsType(IsType { arg, ty });
        self.build(data, Type::Term(TermType::Bool), span)
    }

    #[allow(non_snake_case)]
    fn Binary(
        self,
        op: Opcode,
        ty: Type,
        lhs: Value,
        rhs: Value,
        span: SourceSpan,
    ) -> (Inst, &'f mut DataFlowGraph) {
        let data = InstData::BinaryOp(BinaryOp {
            op,
            args: [lhs, rhs],
        });
        self.build(data, ty, span)
    }

    #[allow(non_snake_case)]
    fn BinaryImm(
        self,
        op: Opcode,
        ty: Type,
        arg: Value,
        imm: Immediate,
        span: SourceSpan,
    ) -> (Inst, &'f mut DataFlowGraph) {
        let data = InstData::BinaryOpImm(BinaryOpImm { op, arg, imm });
        self.build(data, ty, span)
    }

    #[allow(non_snake_case)]
    fn BinaryConst(
        self,
        op: Opcode,
        ty: Type,
        arg: Value,
        imm: Constant,
        span: SourceSpan,
    ) -> (Inst, &'f mut DataFlowGraph) {
        let data = InstData::BinaryOpConst(BinaryOpConst { op, arg, imm });
        self.build(data, ty, span)
    }

    #[allow(non_snake_case)]
    fn Unary(
        self,
        op: Opcode,
        ty: Type,
        arg: Value,
        span: SourceSpan,
    ) -> (Inst, &'f mut DataFlowGraph) {
        let data = InstData::UnaryOp(UnaryOp { op, arg });
        self.build(data, ty, span)
    }

    #[allow(non_snake_case)]
    fn UnaryImm(
        self,
        op: Opcode,
        ty: Type,
        imm: Immediate,
        span: SourceSpan,
    ) -> (Inst, &'f mut DataFlowGraph) {
        let data = InstData::UnaryOpImm(UnaryOpImm { op, imm });
        self.build(data, ty, span)
    }

    #[allow(non_snake_case)]
    fn UnaryConst(
        self,
        op: Opcode,
        ty: Type,
        imm: Constant,
        span: SourceSpan,
    ) -> (Inst, &'f mut DataFlowGraph) {
        let data = InstData::UnaryOpConst(UnaryOpConst { op, imm });
        self.build(data, ty, span)
    }

    #[allow(non_snake_case)]
    fn PrimOp(
        self,
        op: Opcode,
        ty: Type,
        args: ValueList,
        span: SourceSpan,
    ) -> (Inst, &'f mut DataFlowGraph) {
        let data = InstData::PrimOp(PrimOp { op, args });
        self.build(data, ty, span)
    }

    #[allow(non_snake_case)]
    fn PrimOpImm(
        self,
        op: Opcode,
        ty: Type,
        imm: Immediate,
        args: ValueList,
        span: SourceSpan,
    ) -> (Inst, &'f mut DataFlowGraph) {
        let data = InstData::PrimOpImm(PrimOpImm { op, imm, args });
        self.build(data, ty, span)
    }

    #[allow(non_snake_case)]
    fn BitsMatch(
        self,
        spec: BinaryEntrySpecifier,
        args: ValueList,
        span: SourceSpan,
    ) -> (Inst, &'f mut DataFlowGraph) {
        let ty = match spec {
            BinaryEntrySpecifier::Integer { .. } => Type::Term(TermType::Integer),
            BinaryEntrySpecifier::Float { .. } => Type::Term(TermType::Float),
            BinaryEntrySpecifier::Utf8
            | BinaryEntrySpecifier::Utf16 { .. }
            | BinaryEntrySpecifier::Utf32 { .. } => Type::Term(TermType::Integer),
            BinaryEntrySpecifier::Binary { unit: 8, .. } => Type::Term(TermType::Binary),
            BinaryEntrySpecifier::Binary { .. } => Type::Term(TermType::Bitstring),
        };
        let data = InstData::BitsMatch(BitsMatch { spec, args });
        self.build(data, ty, span)
    }

    #[allow(non_snake_case)]
    fn BitsPush(
        self,
        spec: BinaryEntrySpecifier,
        args: ValueList,
        span: SourceSpan,
    ) -> (Inst, &'f mut DataFlowGraph) {
        let data = InstData::BitsPush(BitsPush { spec, args });
        self.build(data, Type::Term(TermType::Any), span)
    }

    #[allow(non_snake_case)]
    fn SetElement(
        self,
        op: Opcode,
        tuple: Value,
        index: Immediate,
        value: Value,
        span: SourceSpan,
    ) -> (Inst, &'f mut DataFlowGraph) {
        let data = InstData::SetElement(SetElement {
            op,
            index,
            args: [tuple, value],
        });
        self.build(data, Type::Term(TermType::Tuple(None)), span)
    }

    #[allow(non_snake_case)]
    fn SetElementImm(
        self,
        op: Opcode,
        tuple: Value,
        index: Immediate,
        value: Immediate,
        span: SourceSpan,
    ) -> (Inst, &'f mut DataFlowGraph) {
        let data = InstData::SetElementImm(SetElementImm {
            op,
            arg: tuple,
            index,
            value,
        });
        self.build(data, Type::Term(TermType::Tuple(None)), span)
    }

    #[allow(non_snake_case)]
    fn SetElementConst(
        self,
        tuple: Value,
        index: Immediate,
        value: Constant,
        span: SourceSpan,
    ) -> (Inst, &'f mut DataFlowGraph) {
        let data = InstData::SetElementConst(SetElementConst {
            op: Opcode::SetElement,
            arg: tuple,
            index,
            value,
        });
        self.build(data, Type::Term(TermType::Tuple(None)), span)
    }

    #[allow(non_snake_case)]
    fn MapUpdate(
        self,
        op: Opcode,
        map: Value,
        key: Value,
        value: Value,
        span: SourceSpan,
    ) -> (Inst, &'f mut DataFlowGraph) {
        let data = InstData::MapUpdate(MapUpdate {
            op,
            args: [map, key, value],
        });
        self.build(data, Type::Term(TermType::Map), span)
    }
}

impl<'f, T: InstBuilderBase<'f>> InstBuilder<'f> for T {}
