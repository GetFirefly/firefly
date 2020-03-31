use thiserror::Error;

#[derive(Debug, Error)]
pub enum TryPropListFromTermError {
    #[error("atom name is not a supported property")]
    AtomName(&'static str),
    #[error("tuple in proplist can only be a 2-tuple")]
    TupleNotPair,
    #[error("a keyword key can only be an atom")]
    KeywordKeyType,
    #[error("keyword key is not a property name")]
    KeywordKeyName(&'static str),
    #[error("property must be a keyword key or an atom")]
    PropertyType,
}
