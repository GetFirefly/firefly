use core::fmt::{self, Debug};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Exception {
    Alloc(Alloc),
}

impl From<core::convert::Infallible> for Exception {
    fn from(_: core::convert::Infallible) -> Self {
        unreachable!()
    }
}

impl From<Alloc> for Exception {
    fn from(alloc: Alloc) -> Self {
        Exception::Alloc(alloc)
    }
}

#[derive(Clone, Copy)]
pub struct Alloc {
    pub file: &'static str,
    pub line: u32,
    pub column: u32,
}

impl Debug for Alloc {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Alloc at {}:{}:{}", self.file, self.line, self.column)
    }
}

impl Eq for Alloc {}

impl PartialEq for Alloc {
    /// `file`, `line`, and `column` don't count for equality as they are for `Debug` only to help
    /// track down exceptions.
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
