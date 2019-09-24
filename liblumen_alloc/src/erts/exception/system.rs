use core::fmt::{self, Debug};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Exception {
    Alloc(Alloc),
}

impl From<Alloc> for Exception {
    fn from(alloc: Alloc) -> Exception {
        Exception::Alloc(alloc)
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
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
