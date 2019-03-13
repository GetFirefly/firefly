use std::fmt::{self, Debug};

pub struct BadArgument {
    pub file: &'static str,
    pub line: u32,
    pub column: u32,
}

impl Debug for BadArgument {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "bad argument")
    }
}

#[macro_export]
macro_rules! bad_argument {
    () => {
        BadArgument {
            file: file!(),
            line: line!(),
            column: column!(),
        }
    };
}
