use super::*;

use libeir_ir::LogicOp;

pub struct LogicOpBuilder;

impl LogicOpBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        ir_value: Option<ir::Value>,
        op: LogicalOperator,
    ) -> Result<Option<Value>> {
        let builder_ref = builder.as_ref();
        let lhs_ref = builder.value_ref(op.lhs);
        let rhs_ref = op.rhs.map(|v| builder.value_ref(v)).unwrap_or_default();

        let result_ref = match op.kind {
            LogicOp::And => unsafe { MLIRBuildLogicalAndOp(builder_ref, op.loc, lhs_ref, rhs_ref) },
            LogicOp::Or => unsafe { MLIRBuildLogicalOrOp(builder_ref, op.loc, lhs_ref, rhs_ref) },
            LogicOp::Eq => todo!("logical primop (eq)"),
        };
        assert!(!result_ref.is_null());

        let result = builder.new_value(ir_value, result_ref, ValueDef::Result(0));
        Ok(Some(result))
    }
}
