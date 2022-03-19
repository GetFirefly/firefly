use liblumen_diagnostics::SourceSpan;

use cranelift_entity::{self as entity, entity_impl};

pub type AnnotationList = entity::EntityList<Annotation>;
pub type AnnotationListPool = entity::ListPool<Annotation>;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Annotation(u32);
entity_impl!(Annotation, "anno");
impl Annotation {
    pub const COMPILER_GENERATED: Self = Self(0);
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnnotationData {
    CompilerGenerated,
    Span(SourceSpan),
}
impl AnnotationData {
    pub fn is_compiler_generated(&self) -> bool {
        match self {
            Self::CompilerGenerated => true,
            _ => false,
        }
    }

    pub fn span(&self) -> Option<SourceSpan> {
        match self {
            Self::Span(span) => Some(*span),
            _ => None,
        }
    }
}

pub struct Annotations<'a> {
    pub(super) inner: entity::Iter<'a, Annotation, AnnotationData>,
}
impl<'a> Iterator for Annotations<'a> {
    type Item = Annotation;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.by_ref().next().map(|kv| kv.0)
    }
}
