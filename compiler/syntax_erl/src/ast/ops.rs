/// The set of all binary operators which may be used in expressions
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum BinaryOp {
    // 100 !, right associative
    Send,
    // 150 orelse
    OrElse,
    // 160 andalso
    AndAlso,
    // 200 <all comparison operators>
    Equal, // right associative
    NotEqual,
    Lte,
    Lt,
    Gte,
    Gt,
    StrictEqual,
    StrictNotEqual,
    // 300 <all list operators>, right associative
    Append,
    Remove,
    // 400 <all add operators>, left associative
    Add,
    Sub,
    Bor,
    Bxor,
    Bsl,
    Bsr,
    Or,
    Xor,
    // 500 <all mul operators>, left associative
    Divide,
    Multiply,
    Div,
    Rem,
    Band,
    And,
}
impl BinaryOp {
    pub fn is_valid_in_patterns(&self) -> bool {
        match self {
            Self::Append
            | Self::Add
            | Self::Sub
            | Self::Multiply
            | Self::Divide
            | Self::Div
            | Self::Rem
            | Self::Band
            | Self::Bor
            | Self::Bxor
            | Self::Bsl
            | Self::Bsr => true,
            _ => false,
        }
    }
}

/// The set of all unary (prefix) operators which may be used in expressions
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum UnaryOp {
    // 600 <all prefix operators>
    Plus,
    Minus,
    Bnot,
    Not,
}
impl UnaryOp {
    pub fn is_valid_in_patterns(&self) -> bool {
        match self {
            Self::Plus | Self::Minus | Self::Bnot => true,
            _ => false,
        }
    }
}
