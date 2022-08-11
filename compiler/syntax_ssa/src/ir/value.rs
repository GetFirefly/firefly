use cranelift_entity::{self as entity, entity_impl};
use liblumen_diagnostics::SourceSpan;

use super::{Block, Inst, Type};

pub type ValueList = entity::EntityList<Value>;
pub type ValueListPool = entity::ListPool<Value>;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Value(u32);
entity_impl!(Value, "v");

#[derive(Debug, Clone)]
pub enum ValueData {
    Inst {
        ty: Type,
        num: u16,
        inst: Inst,
    },
    Param {
        ty: Type,
        num: u16,
        block: Block,
        span: SourceSpan,
    },
}
impl ValueData {
    pub fn ty(&self) -> Type {
        match self {
            Self::Inst { ty, .. } | Self::Param { ty, .. } => ty.clone(),
        }
    }
}

pub struct Values<'a> {
    pub(super) inner: entity::Iter<'a, Value, ValueData>,
}
impl<'a> Iterator for Values<'a> {
    type Item = Value;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.by_ref().next().map(|kv| kv.0)
    }
}
