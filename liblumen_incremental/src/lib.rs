mod intern;
mod queries;
mod query_groups;

pub use self::intern::InternedInput;
pub use self::query_groups::*;

pub type QueryResult<T> = Result<T, ()>;
