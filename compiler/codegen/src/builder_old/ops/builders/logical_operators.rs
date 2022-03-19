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
        let operand_refs = op
            .operands
            .iter()
            .copied()
            .map(|value| builder.value_ref(value))
            .collect::<Vec<_>>();
        builder.debug(&format!(
            "logic operation ({:?}) operands: {:?}",
            op.kind,
            operand_refs.as_slice()
        ));

        let result_ref = match op.kind {
            LogicOp::And => unsafe {
                MLIRBuildLogicalAndOp(
                    builder_ref,
                    op.loc,
                    operand_refs.as_ptr(),
                    operand_refs.len() as c_uint,
                )
            },
            LogicOp::Or => unsafe {
                MLIRBuildLogicalOrOp(
                    builder_ref,
                    op.loc,
                    operand_refs.as_ptr(),
                    operand_refs.len() as c_uint,
                )
            },
            LogicOp::Eq => todo!("logical primop (eq)"),
        };
        assert!(!result_ref.is_null());

        let result = builder.new_value(ir_value, result_ref, ValueDef::Result(0));
        Ok(Some(result))
    }
}
