/// Represents original source file location information present in Erlang Abstract Format
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Loc {
    line: u32,
    column: u32,
}
impl Loc {
    pub fn new(line: u32, column: u32) -> Self {
        Self { line, column }
    }
}
