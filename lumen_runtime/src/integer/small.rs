use crate::term::Tag;

#[cfg_attr(test, derive(Debug))]
pub struct Integer(pub isize);

pub const MIN: isize = std::isize::MIN >> Tag::SMALL_INTEGER_BIT_COUNT;
pub const MAX: isize = std::isize::MAX >> Tag::SMALL_INTEGER_BIT_COUNT;
