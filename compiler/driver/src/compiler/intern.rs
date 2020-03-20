#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct InternedString(salsa::InternId);

impl salsa::InternKey for InternedString {
    fn from_intern_id(id: salsa::InternId) -> Self {
        Self(id)
    }

    fn as_intern_id(&self) -> salsa::InternId {
        self.0
    }
}
