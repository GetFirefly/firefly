use std::fmt;

#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(C)]
pub enum Dialect {
    #[allow(dead_code)]
    Other,
    None,
    EIR,
    Standard,
    LLVM,
}
impl fmt::Display for Dialect {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut name = format!("{:?}", self);
        name.make_ascii_lowercase();
        write!(f, "{}", &name)
    }
}
