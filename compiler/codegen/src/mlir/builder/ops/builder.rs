use std::convert::From;

use anyhow::anyhow;
use libeir_ir as ir;

use crate::mlir::builder::value::Value;
use crate::mlir::builder::ScopedFunctionBuilder;
use crate::Result;

use super::builders::*;
use super::OpKind;

pub struct OpBuilder;

impl OpBuilder {
    #[inline]
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        ir_value: Option<ir::Value>,
        kind: OpKind,
    ) -> Result<Option<Value>> {
        Self::do_build(builder, ir_value, kind)
    }

    pub fn build_void_result<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        kind: OpKind,
    ) -> Result<()> {
        if let Some(_) = Self::do_build(builder, None, kind)? {
            Err(anyhow!("expected operation to return void"))
        } else {
            Ok(())
        }
    }

    pub fn build_one_result<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        ir_value: ir::Value,
        kind: OpKind,
    ) -> Result<Value> {
        if let Some(value) = Self::do_build(builder, Some(ir_value), kind)? {
            Ok(value)
        } else {
            Err(anyhow!("expected operation to return a value"))
        }
    }

    fn do_build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        ir_value: Option<ir::Value>,
        kind: OpKind,
    ) -> Result<Option<Value>> {
        match kind {
            OpKind::Return(value) => ReturnBuilder::build(builder, value),
            OpKind::Throw(op) => ThrowBuilder::build(builder, op),
            OpKind::Unreachable => UnreachableBuilder::build(builder),
            OpKind::Call(call) => CallBuilder::build(builder, ir_value, call),
            OpKind::Branch(branch) => BranchBuilder::build(builder, branch),
            OpKind::If(op) => IfBuilder::build(builder, op),
            OpKind::IsType { value, expected } => {
                IsTypeBuilder::build(builder, ir_value, value, expected)
            }
            OpKind::Match(op) => MatchBuilder::build(builder, op),
            OpKind::BinOp(op) => BinOpBuilder::build(builder, ir_value, op),
            OpKind::LogicOp(op) => LogicOpBuilder::build(builder, ir_value, op),
            OpKind::Constant(c) => ConstantBuilder::build(builder, ir_value, c),
            OpKind::FunctionRef(callee) => CalleeBuilder::build(builder, ir_value, callee),
            OpKind::Tuple(elements) => TupleBuilder::build(builder, ir_value, elements.as_slice()),
            OpKind::Cons(head, tail) => ConsBuilder::build(builder, ir_value, head, tail),
            OpKind::Map(items) => MapBuilder::build(builder, ir_value, items.as_slice()),
            OpKind::MapPut(op) => MapPutBuilder::build(builder, op),
            OpKind::BinaryPush(op) => BinaryPushBuilder::build(builder, op),
            OpKind::TraceCapture(branch) => TraceCaptureBuilder::build(builder, branch),
            OpKind::TraceConstruct(capture) => {
                TraceConstructBuilder::build(builder, ir_value, capture)
            }
            OpKind::Intrinsic(op) => IntrinsicBuilder::build(builder, ir_value, op),
        }
    }
}
