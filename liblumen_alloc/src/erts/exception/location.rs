use core::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Location {
    file: &'static str,
    line: u32,
    column: u32,
}
impl Location {
    #[inline]
    pub fn new(file: &'static str, line: u32, column: u32) -> Self {
        Self {
            file,
            line,
            column,
        }
    }
}
impl Default for Location {
    fn default() -> Self {
        Self::new("nofile", 0, 0)
    }
}
impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} at {}:{}", self.file, self.file, self.column)
    }
}
