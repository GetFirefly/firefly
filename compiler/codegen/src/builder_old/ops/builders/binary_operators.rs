use super::*;

pub struct BinOpBuilder;

impl BinOpBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        ir_value: Option<ir::Value>,
        op: BinaryOperator,
    ) -> Result<Option<Value>> {
        let builder_ref = builder.as_ref();
        let lhs_ref = builder.value_ref(op.lhs);
        let rhs_ref = builder.value_ref(op.rhs);

        let result_ref = match op.kind {
            ir::BinOp::Equal => unsafe {
                MLIRBuildIsEqualOp(
                    builder_ref,
                    op.loc,
                    lhs_ref,
                    rhs_ref,
                    /* isExact= */ false,
                )
            },
            ir::BinOp::ExactEqual => unsafe {
                MLIRBuildIsEqualOp(
                    builder_ref,
                    op.loc,
                    lhs_ref,
                    rhs_ref,
                    /* isExact= */ true,
                )
            },
            ir::BinOp::NotEqual => unsafe {
                MLIRBuildIsNotEqualOp(
                    builder_ref,
                    op.loc,
                    lhs_ref,
                    rhs_ref,
                    /* isExact= */ false,
                )
            },
            ir::BinOp::ExactNotEqual => unsafe {
                MLIRBuildIsNotEqualOp(
                    builder_ref,
                    op.loc,
                    lhs_ref,
                    rhs_ref,
                    /* isExact= */ true,
                )
            },
            ir::BinOp::LessEqual => unsafe {
                MLIRBuildLessThanOrEqualOp(builder_ref, op.loc, lhs_ref, rhs_ref)
            },
            ir::BinOp::Less => unsafe {
                MLIRBuildLessThanOp(builder_ref, op.loc, lhs_ref, rhs_ref)
            },
            ir::BinOp::GreaterEqual => unsafe {
                MLIRBuildGreaterThanOrEqualOp(builder_ref, op.loc, lhs_ref, rhs_ref)
            },
            ir::BinOp::Greater => unsafe {
                MLIRBuildGreaterThanOp(builder_ref, op.loc, lhs_ref, rhs_ref)
            },
        };
        assert!(!result_ref.is_null());

        let result = builder.new_value(ir_value, result_ref, ValueDef::Result(0));
        Ok(Some(result))
    }
}
