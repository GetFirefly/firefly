use liblumen_intern::{symbols, Symbol};

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
    pub fn from_symbol(sym: Symbol) -> Result<Self, ()> {
        let op = match sym {
            symbols::Bang => Self::Send,
            symbols::OrElse => Self::OrElse,
            symbols::AndAlso => Self::AndAlso,
            symbols::Equal => Self::Equal,
            symbols::NotEqual => Self::NotEqual,
            symbols::Lte => Self::Lte,
            symbols::Lt => Self::Lt,
            symbols::Gte => Self::Gte,
            symbols::Gt => Self::Gt,
            symbols::EqualStrict => Self::StrictEqual,
            symbols::NotEqualStrict => Self::StrictNotEqual,
            symbols::PlusPlus => Self::Append,
            symbols::MinusMinus => Self::Remove,
            symbols::Plus => Self::Add,
            symbols::Minus => Self::Sub,
            symbols::Bor => Self::Bor,
            symbols::Bxor => Self::Bxor,
            symbols::Bsl => Self::Bsl,
            symbols::Bsr => Self::Bsr,
            symbols::Or => Self::Or,
            symbols::Xor => Self::Xor,
            symbols::Slash => Self::Divide,
            symbols::Star => Self::Multiply,
            symbols::Div => Self::Div,
            symbols::Rem => Self::Rem,
            symbols::Band => Self::Band,
            symbols::And => Self::And,
            _ => return Err(()),
        };
        Ok(op)
    }

    pub fn to_symbol(&self) -> Symbol {
        match self {
            Self::Send => symbols::Bang,
            Self::OrElse => symbols::OrElse,
            Self::AndAlso => symbols::AndAlso,
            Self::Equal => symbols::Equal,
            Self::NotEqual => symbols::NotEqual,
            Self::Lte => symbols::Lte,
            Self::Lt => symbols::Lt,
            Self::Gte => symbols::Gte,
            Self::Gt => symbols::Gt,
            Self::StrictEqual => symbols::EqualStrict,
            Self::StrictNotEqual => symbols::NotEqualStrict,
            Self::Append => symbols::PlusPlus,
            Self::Remove => symbols::MinusMinus,
            Self::Add => symbols::Plus,
            Self::Sub => symbols::Minus,
            Self::Bor => symbols::Bor,
            Self::Bxor => symbols::Bxor,
            Self::Bsl => symbols::Bsl,
            Self::Bsr => symbols::Bsr,
            Self::Or => symbols::Or,
            Self::Xor => symbols::Xor,
            Self::Divide => symbols::Slash,
            Self::Multiply => symbols::Star,
            Self::Div => symbols::Div,
            Self::Rem => symbols::Rem,
            Self::Band => symbols::Band,
            Self::And => symbols::And,
        }
    }

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

    pub fn is_guard_op(&self) -> bool {
        match self {
            Self::OrElse
            | Self::AndAlso
            | Self::Equal
            | Self::NotEqual
            | Self::Lte
            | Self::Lt
            | Self::Gte
            | Self::Gt
            | Self::StrictEqual
            | Self::StrictNotEqual
            | Self::Add
            | Self::Sub
            | Self::Bor
            | Self::Bxor
            | Self::Bsl
            | Self::Bsr
            | Self::Or
            | Self::Xor
            | Self::Divide
            | Self::Multiply
            | Self::Div
            | Self::Rem
            | Self::Band
            | Self::And => true,
            _ => false,
        }
    }

    // See erl_internal:bool_op/2
    #[inline]
    pub fn is_boolean(&self, arity: u8) -> bool {
        is_boolean_op(self.to_symbol(), arity)
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
    pub fn from_symbol(sym: Symbol) -> Result<Self, ()> {
        let op = match sym {
            symbols::Plus => Self::Plus,
            symbols::Minus => Self::Minus,
            symbols::Bnot => Self::Bnot,
            symbols::Not => Self::Not,
            _ => return Err(()),
        };
        Ok(op)
    }

    pub fn to_symbol(&self) -> Symbol {
        match self {
            Self::Plus => symbols::Plus,
            Self::Minus => symbols::Minus,
            Self::Bnot => symbols::Bnot,
            Self::Not => symbols::Not,
        }
    }

    pub fn is_valid_in_patterns(&self) -> bool {
        match self {
            Self::Plus | Self::Minus | Self::Bnot => true,
            _ => false,
        }
    }

    pub fn is_guard_op(&self) -> bool {
        true
    }

    pub fn is_boolean(&self, arity: u8) -> bool {
        is_boolean_op(self.to_symbol(), arity)
    }
}

pub fn is_boolean_op(sym: Symbol, arity: u8) -> bool {
    match sym {
        symbols::Not if arity == 1 => true,
        symbols::Or | symbols::And | symbols::Xor if arity == 2 => true,
        _ => false,
    }
}
