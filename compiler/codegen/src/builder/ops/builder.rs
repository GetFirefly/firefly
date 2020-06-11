use anyhow::anyhow;
use libeir_ir as ir;

use crate::builder::value::Value;
use crate::builder::ScopedFunctionBuilder;
use crate::Result;

use super::builders::*;
use super::OpKind;

pub struct OpBuilder;

impl OpBuilder {
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
            OpKind::Return(op) => ReturnBuilder::build(builder, op),
            OpKind::Throw(op) => ThrowBuilder::build(builder, op),
            OpKind::Unreachable(loc) => UnreachableBuilder::build(builder, loc),
            OpKind::Call(op) => CallBuilder::build(builder, ir_value, op),
            OpKind::Branch(op) => BranchBuilder::build(builder, op),
            OpKind::If(op) => IfBuilder::build(builder, op),
            OpKind::IsType(op) => IsTypeBuilder::build(builder, ir_value, op),
            OpKind::Match(op) => MatchBuilder::build(builder, op),
            OpKind::BinOp(op) => BinOpBuilder::build(builder, ir_value, op),
            OpKind::LogicOp(op) => LogicOpBuilder::build(builder, ir_value, op),
            OpKind::Constant(op) => ConstantBuilder::build(builder, ir_value, op),
            OpKind::FunctionRef(op) => CalleeBuilder::build(builder, ir_value, op),
            OpKind::Tuple(op) => TupleBuilder::build(builder, ir_value, op),
            OpKind::Cons(op) => ConsBuilder::build(builder, ir_value, op),
            OpKind::Map(op) => MapBuilder::build(builder, ir_value, op),
            OpKind::MapPut(op) => MapPutBuilder::build(builder, op),
            OpKind::TraceCapture(op) => TraceCaptureBuilder::build(builder, op),
            OpKind::TraceConstruct(op) => TraceConstructBuilder::build(builder, ir_value, op),
            OpKind::BinaryStart(op) => BinaryStartBuilder::build(builder, op),
            OpKind::BinaryPush(op) => BinaryPushBuilder::build(builder, op),
            OpKind::BinaryFinish(op) => BinaryFinishBuilder::build(builder, op),
            OpKind::ReceiveStart(op) => ReceiveStartBuilder::build(builder, op),
            OpKind::ReceiveWait(op) => ReceiveWaitBuilder::build(builder, op),
            OpKind::ReceiveDone(op) => ReceiveDoneBuilder::build(builder, op),
        }
    }
}
