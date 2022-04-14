use std::fmt;

use super::FuncRef;

/// Concrete data about a specific type instance
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Invalid,
    Term,
    Bool,
    Integer,
    Float,
    Number,
    Atom,
    Bitstring,
    Binary,
    Nil,
    List(Option<Box<Type>>),
    MaybeImproperList,
    Tuple(Option<Vec<Type>>),
    Map(Option<(Box<Type>, Box<Type>)>),
    Reference,
    Port,
    Pid,
    Fun(Option<FuncRef>),
    NoReturn,
    // This type maps to ErlangException in liblumen_alloc
    Exception,
    // This type maps to Trace in liblumen_alloc
    ExceptionTrace,
    // This type maps to ReceiveContext in lumen_rt_minimal
    RecvContext,
    // This type maps to ReceiveState in lumen_rt_minimal
    RecvState,
}
impl Type {
    pub fn tuple(arity: usize) -> Type {
        let mut elements = Vec::with_capacity(arity);
        elements.resize(arity, Self::Term);
        Self::Tuple(Some(elements))
    }

    pub fn is_opaque(&self) -> bool {
        match self {
            Self::Term => true,
            _ => false,
        }
    }

    pub fn is_numeric(&self) -> bool {
        match self {
            Self::Integer | Self::Float | Self::Number => true,
            _ => false,
        }
    }

    pub fn is_atom(&self) -> bool {
        match self {
            Self::Bool | Self::Atom => true,
            _ => false,
        }
    }

    pub fn is_list(&self) -> bool {
        match self {
            Self::List(_) | Self::MaybeImproperList | Self::Nil => true,
            _ => false,
        }
    }

    pub fn is_tuple(&self) -> bool {
        match self {
            Self::Tuple(_) => true,
            _ => false,
        }
    }

    pub fn is_map(&self) -> bool {
        match self {
            Self::Map(_) => true,
            _ => false,
        }
    }

    pub fn is_bitstring(&self) -> bool {
        match self {
            Self::Bitstring | Self::Binary => true,
            _ => false,
        }
    }

    /// If we have to coerce this to the most precise numeric type we can, what type would that be?
    pub fn coerce_to_numeric(&self) -> Self {
        match self {
            Self::Integer => Self::Integer,
            Self::Float => Self::Float,
            Self::Number => Self::Number,
            // We assume this is just due to a lack of type information
            Self::Term => Self::Number,
            _ => Self::Invalid,
        }
    }

    /// If we have a binary arithemtic operation, what is the type of the result based on the type of the operands?
    pub fn coerce_to_numeric_with(&self, other: Self) -> Self {
        let this = self.coerce_to_numeric();
        let other = other.coerce_to_numeric();
        // If the types are the same, we have our answer
        if this == other {
            return this;
        }
        // Otherwise, coerce to the appropriate type for binary arithmetic ops
        match (this, other) {
            // Invalid operands are guaranteed to fail, so we propagate an invalid type
            (Self::Invalid, _) => Self::Invalid,
            (_, Self::Invalid) => Self::Invalid,
            // Mixed integer/float ops always produce floats
            _ => Self::Float,
        }
    }
}
impl fmt::Display for Type {
    /// Print this type for display using the provided module context
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use std::fmt::Write;
        match self {
            Self::Invalid => f.write_str("invalid"),
            Self::Term => f.write_str("term"),
            Self::Bool => f.write_str("bool"),
            Self::Integer => f.write_str("int"),
            Self::Float => f.write_str("float"),
            Self::Number => f.write_str("number"),
            Self::Atom => f.write_str("atom"),
            Self::Bitstring => f.write_str("bits"),
            Self::Binary => f.write_str("bytes"),
            Self::Nil => f.write_str("nil"),
            Self::List(None) => f.write_str("list"),
            Self::List(Some(ty)) => write!(f, "list<{}>", ty.as_ref()),
            Self::MaybeImproperList => f.write_str("list?"),
            Self::Tuple(None) => f.write_str("tuple"),
            Self::Tuple(Some(ref elements)) => {
                f.write_str("tuple<")?;
                for (i, t) in elements.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}", t)?;
                }
                f.write_str(">")
            }
            Self::Map(None) => f.write_str("map"),
            Self::Map(Some((k, v))) => write!(f, "map<{}, {}>", k, v),
            Self::Reference => f.write_str("reference"),
            Self::Port => f.write_str("port"),
            Self::Pid => f.write_str("pid"),
            Self::Fun(None) => f.write_str("fun"),
            Self::Fun(Some(func_ref)) => write!(f, "fun({})", func_ref),
            Self::NoReturn => f.write_char('!'),
            Self::Exception => f.write_str("exception"),
            Self::ExceptionTrace => f.write_str("trace"),
            Self::RecvContext => f.write_str("recv_context"),
            Self::RecvState => f.write_str("recv_state"),
        }
    }
}
impl PartialOrd for Type {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering::*;

        // Invalid/NoReturn cannot be compared
        if self == &Self::Invalid || other == &Self::Invalid {
            return None;
        }
        if self == &Self::NoReturn || other == &Self::NoReturn {
            return None;
        }

        // If the types are the same, they are equal, unless they are
        // terms in which case they cannot be compared
        if self == other {
            if self == &Self::Term {
                return None;
            }
            return Some(Equal);
        }

        // We try to follow the term comparison order here, but we're not
        // trying too hard, e.g. we consider tuples with the same element
        // types to be equivalent. This simply aids us in resolving comparisons
        // at compile-time if they are known to be less-than/greater-than
        // the other operand
        match (self, other) {
            (Self::Term, _) => None,
            (_, Self::Term) => None,
            (Self::Integer, Self::Float) => Some(Equal),
            (Self::Float, Self::Integer) => Some(Equal),
            (Self::Integer, _) => Some(Less),
            (Self::Float, _) => Some(Less),
            (_, Self::Integer) => Some(Greater),
            (_, Self::Float) => Some(Greater),
            (Self::Number, _) => Some(Less),
            (_, Self::Number) => Some(Greater),
            (Self::Bool, Self::Atom) => Some(Equal),
            (Self::Atom, Self::Bool) => Some(Equal),
            (Self::Bool, _) => Some(Less),
            (_, Self::Bool) => Some(Greater),
            (Self::Atom, _) => Some(Less),
            (_, Self::Atom) => Some(Greater),
            (Self::Reference, _) => Some(Less),
            (_, Self::Reference) => Some(Greater),
            (Self::Fun(_), _) => Some(Less),
            (_, Self::Fun(_)) => Some(Greater),
            (Self::Port, _) => Some(Less),
            (_, Self::Port) => Some(Greater),
            (Self::Pid, _) => Some(Less),
            (_, Self::Pid) => Some(Greater),
            (Self::Tuple(_), _) => Some(Less),
            (_, Self::Tuple(_)) => Some(Greater),
            (Self::Map(_), _) => Some(Less),
            (_, Self::Map(_)) => Some(Greater),
            (Self::Nil, _) => Some(Less),
            (_, Self::Nil) => Some(Greater),
            (Self::List(_), _) => Some(Less),
            (_, Self::List(_)) => Some(Greater),
            (Self::MaybeImproperList, _) => Some(Less),
            (_, Self::MaybeImproperList) => Some(Greater),
            (Self::Bitstring, Self::Binary) => Some(Equal),
            (Self::Binary, Self::Bitstring) => Some(Equal),
            (Self::Bitstring, _) => Some(Less),
            (Self::Binary, _) => Some(Less),
            (_, _) => None,
        }
    }
}
