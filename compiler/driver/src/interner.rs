use firefly_session::Input;

/// Maps to an interned instance of Input
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct InternedInput(salsa::InternId);
impl From<u32> for InternedInput {
    #[inline]
    fn from(i: u32) -> Self {
        Self(i.into())
    }
}
impl salsa::InternKey for InternedInput {
    fn from_intern_id(id: salsa::InternId) -> Self {
        Self(id)
    }

    fn as_intern_id(&self) -> salsa::InternId {
        self.0
    }
}

#[salsa::query_group(InternerStorage)]
pub trait Interner: salsa::Database {
    #[salsa::interned]
    fn intern_input(&self, input: Input) -> InternedInput;
}
