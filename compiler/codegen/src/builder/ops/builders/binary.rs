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
        let head = builder.value_ref(op.head);
        let tail = builder.value_ref(op.tail);
        let size = op.size.map(|s| builder.value_ref(s)).unwrap_or_default();
        let spec = (&op.spec).into();

        unsafe {
            MLIRBuildBinaryPush(builder.as_ref(), op.loc, head, tail, size, &spec, ok, err);
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
        let cont = builder.block_ref(op.cont);
        let bin = builder.value_ref(op.bin);

        unsafe {
            MLIRBuildBinaryFinish(builder.as_ref(), op.loc, cont, bin);
        }

        Ok(None)
    }
}
