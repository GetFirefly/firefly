use super::*;

pub struct BinaryStartBuilder;

impl BinaryStartBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        op: BinaryStart,
    ) -> Result<Option<Value>> {
        let cont = builder.block_ref(op.cont);

        unsafe {
            MLIRBuildBinaryStart(builder.as_ref(), op.loc, cont);
        }

        Ok(None)
    }
}

pub struct BinaryPushBuilder;

impl BinaryPushBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        op: BinaryPush,
    ) -> Result<Option<Value>> {
        let ok = builder.block_ref(op.ok);
        let err = builder.block_ref(op.err);
        let bin = builder.value_ref(op.bin);
        let value = builder.value_ref(op.value);
        let size = op.size.map(|s| builder.value_ref(s)).unwrap_or_default();
        let spec = (&op.spec).into();

        unsafe {
            MLIRBuildBinaryPush(builder.as_ref(), op.loc, bin, value, size, &spec, ok, err);
        }

        Ok(None)
    }
}

pub struct BinaryFinishBuilder;

impl BinaryFinishBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        op: BinaryFinish,
    ) -> Result<Option<Value>> {
        let cont = op.cont.map(|b| builder.block_ref(b)).unwrap_or_default();
        let bin = builder.value_ref(op.bin);

        unsafe {
            MLIRBuildBinaryFinish(builder.as_ref(), op.loc, cont, bin);
        }

        Ok(None)
    }
}
