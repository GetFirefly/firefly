use super::*;

pub struct BinaryPushBuilder;

impl BinaryPushBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        op: BinaryPush,
    ) -> Result<Option<Value>> {
        todo!("build binary push {:#?}", op);
    }
}
