//! Token values.

/// Keyword (a.k.a., reserved word).
///
/// Reference: [Erlang's Reserved Words][Reserved Words]
///
/// [Reserved Words]: http://erlang.org/doc/reference_manual/introduction.html#id61721
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Keyword {
    /// `after`
    After,

    /// `and`
    And,

    /// `andalso`
    Andalso,

    /// `band`
    Band,

    /// `begin`
    Begin,

    /// `bnot`
    Bnot,

    /// `bor`
    Bor,

    /// `bsl`
    Bsl,

    /// `bsr`
    Bsr,

    /// `bxor`
    Bxor,

    /// `case`
    Case,

    /// `catch`
    Catch,

    /// `cond`
    Cond,

    /// `div`
    Div,

    /// `end`
    End,

    /// `fun`
    Fun,

    /// `if`
    If,

    /// `let`
    Let,

    /// `not`
    Not,

    /// `of`
    Of,

    /// `or`
    Or,

    /// `orelse`
    Orelse,

    /// `receive`
    Receive,

    /// `rem`
    Rem,

    /// `try`
    Try,

    /// `when`
    When,

    /// `xor`
    Xor,
}
impl Keyword {
    /// Returns the string representation of this keyword.
    pub fn as_str(&self) -> &'static str {
        match *self {
            Keyword::After => "after",
            Keyword::And => "and",
            Keyword::Andalso => "andalso",
            Keyword::Band => "band",
            Keyword::Begin => "begin",
            Keyword::Bnot => "bnot",
            Keyword::Bor => "bor",
            Keyword::Bsl => "bsl",
            Keyword::Bsr => "bsr",
            Keyword::Bxor => "bxor",
            Keyword::Case => "case",
            Keyword::Catch => "catch",
            Keyword::Cond => "cond",
            Keyword::Div => "div",
            Keyword::End => "end",
            Keyword::Fun => "fun",
            Keyword::If => "if",
            Keyword::Let => "let",
            Keyword::Not => "not",
            Keyword::Of => "of",
            Keyword::Or => "or",
            Keyword::Orelse => "orelse",
            Keyword::Receive => "receive",
            Keyword::Rem => "rem",
            Keyword::Try => "try",
            Keyword::When => "when",
            Keyword::Xor => "xor",
        }
    }
}

/// Symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Symbol {
    /// `[`
    OpenSquare,

    /// `]`
    CloseSquare,

    /// `(`
    OpenParen,

    /// `)`
    CloseParen,

    /// `{`
    OpenBrace,

    /// `}`
    CloseBrace,

    /// `#`
    Sharp,

    /// `/`
    Slash,

    /// `.`
    Dot,

    /// `..`
    DoubleDot,

    /// `...`
    TripleDot,

    /// `,`
    Comma,

    /// `:`
    Colon,

    /// `::`
    DoubleColon,

    /// `;`
    Semicolon,

    /// `=`
    Match,

    /// `:=`
    MapMatch,

    /// `|`
    VerticalBar,

    /// `||`
    DoubleVerticalBar,

    /// `?`
    Question,

    /// `??`
    DoubleQuestion,

    /// `!`
    Not,

    /// `-`
    Hyphen,

    /// `--`
    MinusMinus,

    /// `+`
    Plus,

    /// `++`
    PlusPlus,

    /// `*`
    Multiply,

    /// `->`
    RightArrow,

    /// `<-`
    LeftArrow,

    /// `=>`
    DoubleRightArrow,

    /// `<=`
    DoubleLeftArrow,

    /// `>>`
    DoubleRightAngle,

    /// `<<`
    DoubleLeftAngle,

    /// `==`
    Eq,

    /// `=:=`
    ExactEq,

    /// `/=`
    NotEq,

    /// `=/=`
    ExactNotEq,

    /// `>`
    Greater,

    /// `>=`
    GreaterEq,

    /// `<`
    Less,

    /// `=<`
    LessEq,
}
impl Symbol {
    /// Returns the textual representation of this symbol.
    pub fn as_str(&self) -> &'static str {
        match *self {
            Symbol::OpenSquare => "[",
            Symbol::CloseSquare => "]",
            Symbol::OpenParen => "(",
            Symbol::CloseParen => ")",
            Symbol::OpenBrace => "{",
            Symbol::CloseBrace => "}",
            Symbol::Sharp => "#",
            Symbol::Slash => "/",
            Symbol::Dot => ".",
            Symbol::DoubleDot => "..",
            Symbol::TripleDot => "...",
            Symbol::Comma => ",",
            Symbol::Colon => ":",
            Symbol::DoubleColon => "::",
            Symbol::Semicolon => ";",
            Symbol::Match => "=",
            Symbol::MapMatch => ":=",
            Symbol::VerticalBar => "|",
            Symbol::DoubleVerticalBar => "||",
            Symbol::Question => "?",
            Symbol::DoubleQuestion => "??",
            Symbol::Not => "!",
            Symbol::Hyphen => "-",
            Symbol::MinusMinus => "--",
            Symbol::Plus => "+",
            Symbol::PlusPlus => "++",
            Symbol::Multiply => "*",
            Symbol::RightArrow => "->",
            Symbol::LeftArrow => "<-",
            Symbol::DoubleRightArrow => "=>",
            Symbol::DoubleLeftArrow => "<=",
            Symbol::DoubleRightAngle => ">>",
            Symbol::DoubleLeftAngle => "<<",
            Symbol::Eq => "==",
            Symbol::ExactEq => "=:=",
            Symbol::NotEq => "/=",
            Symbol::ExactNotEq => "=/=",
            Symbol::Greater => ">",
            Symbol::GreaterEq => ">=",
            Symbol::Less => "<",
            Symbol::LessEq => "=<",
        }
    }
}

/// White space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Whitespace {
    /// `' '`
    Space,

    /// `'\t'`
    Tab,

    /// `'\r'`
    Return,

    /// `'\n'`
    Newline,

    /// `'\u{A0}'`
    NoBreakSpace,
}
impl Whitespace {
    /// Coverts to the corresponding character.
    pub fn as_char(&self) -> char {
        match *self {
            Whitespace::Space => ' ',
            Whitespace::Tab => '\t',
            Whitespace::Return => '\r',
            Whitespace::Newline => '\n',
            Whitespace::NoBreakSpace => '\u{A0}',
        }
    }

    /// Coverts to the corresponding string.
    pub fn as_str(&self) -> &'static str {
        match *self {
            Whitespace::Space => " ",
            Whitespace::Tab => "\t",
            Whitespace::Return => "\r",
            Whitespace::Newline => "\n",
            Whitespace::NoBreakSpace => "\u{A0}",
        }
    }
}
