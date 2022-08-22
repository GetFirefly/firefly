use std::fmt::{self, Write};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct FunctionType {
    pub results: Vec<Type>,
    pub params: Vec<Type>,
}
impl FunctionType {
    pub fn new(params: Vec<Type>, results: Vec<Type>) -> Self {
        Self { results, params }
    }

    pub fn arity(&self) -> usize {
        self.params.len()
    }

    pub fn results(&self) -> &[Type] {
        self.results.as_slice()
    }

    pub fn params(&self) -> &[Type] {
        self.params.as_slice()
    }
}
impl fmt::Display for FunctionType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_char('(')?;
        for (i, ty) in self.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", {}", ty)?;
            } else {
                write!(f, "{}", ty)?;
            }
        }
        f.write_str(" -> (")?;
        for (i, ty) in self.results.iter().enumerate() {
            if i > 0 {
                write!(f, ", {}", ty)?;
            } else {
                write!(f, "{}", ty)?;
            }
        }
        f.write_char(')')
    }
}

/// Types in this enumeration correspond to primitive LLVM types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PrimitiveType {
    Void,
    I1,
    I8,
    I16,
    I32,
    I64,
    Isize,
    F64,
    Ptr(Box<PrimitiveType>),
    Struct(Vec<PrimitiveType>),
    Array(Box<PrimitiveType>, usize),
}
impl PrimitiveType {
    pub fn is_integer(&self) -> bool {
        match self {
            Self::I1 | Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::Isize => true,
            _ => false,
        }
    }

    pub fn is_float(&self) -> bool {
        match self {
            Self::F64 => true,
            _ => false,
        }
    }

    pub fn is_pointer(&self) -> bool {
        match self {
            Self::Ptr(_) => true,
            _ => false,
        }
    }

    pub fn is_struct(&self) -> bool {
        match self {
            Self::Struct(_) => true,
            _ => false,
        }
    }

    pub fn is_array(&self) -> bool {
        match self {
            Self::Array(_, _) => true,
            _ => false,
        }
    }
}
impl fmt::Display for PrimitiveType {
    /// Print this type for display using the provided module context
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Void => f.write_str("void"),
            Self::I1 => f.write_str("i1"),
            Self::I8 => f.write_str("i8"),
            Self::I16 => f.write_str("i16"),
            Self::I32 => f.write_str("i32"),
            Self::I64 => f.write_str("i64"),
            Self::Isize => f.write_str("isize"),
            Self::F64 => f.write_str("f64"),
            Self::Ptr(inner) => write!(f, "ptr<{}>", &inner),
            Self::Struct(fields) => {
                f.write_str("{")?;
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", {}", field)?;
                    } else {
                        write!(f, "{}", field)?;
                    }
                }
                f.write_str("}")
            }
            Self::Array(element_ty, arity) => write!(f, "[{}; {}]", &element_ty, arity),
        }
    }
}

/// Types in this enumeration are high-level types that require translation to primitive types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TermType {
    Any,
    Bool,
    Integer,
    Float,
    Number,
    Atom,
    Bitstring,
    Binary,
    /// The empty list
    Nil,
    /// A non-empty list of at least one element
    Cons,
    /// A potentially-empty proper list, with an optional constraint on the element types
    List(Option<Box<TermType>>),
    /// A potentially-empty improper list
    MaybeImproperList,
    Tuple(Option<Vec<TermType>>),
    Map,
    Reference,
    Port,
    Pid,
    Fun(Option<Box<FunctionType>>),
}
impl TermType {
    pub fn is_opaque(&self) -> bool {
        match self {
            Self::Any => true,
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
            Self::List(_) | Self::MaybeImproperList | Self::Cons | Self::Nil => true,
            _ => false,
        }
    }

    pub fn is_nonempty_list(&self) -> bool {
        match self {
            Self::Cons => true,
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
            Self::Map => true,
            _ => false,
        }
    }

    pub fn is_bitstring(&self) -> bool {
        match self {
            Self::Bitstring | Self::Binary => true,
            _ => false,
        }
    }

    pub fn is_fun(&self) -> bool {
        match self {
            Self::Fun(_) => true,
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
            Self::Any => Self::Number,
            other => panic!("invalid type coercion of {} to numeric", other),
        }
    }

    /// If we have a binary arithemtic operation, what is the type of the result based on the type of the operands?
    pub fn coerce_to_numeric_with(&self, other: Self) -> Self {
        let this = self.coerce_to_numeric();
        let other = other.coerce_to_numeric();
        // If the types are the same, we have our answer
        if this == other {
            return this;
        } else {
            // Mixed integer/float ops always produce floats
            Self::Float
        }
    }
}
impl fmt::Display for TermType {
    /// Print this type for display using the provided module context
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Any => f.write_str("term"),
            Self::Bool => f.write_str("bool"),
            Self::Integer => f.write_str("int"),
            Self::Float => f.write_str("float"),
            Self::Number => f.write_str("number"),
            Self::Atom => f.write_str("atom"),
            Self::Bitstring => f.write_str("bits"),
            Self::Binary => f.write_str("bytes"),
            Self::Nil => f.write_str("nil"),
            Self::Cons => f.write_str("cons"),
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
            Self::Map => f.write_str("map"),
            Self::Reference => f.write_str("reference"),
            Self::Port => f.write_str("port"),
            Self::Pid => f.write_str("pid"),
            Self::Fun(None) => f.write_str("fun"),
            Self::Fun(Some(ty)) => write!(f, "fun{}", &ty),
        }
    }
}
impl PartialOrd for TermType {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering::*;

        // If the types are the same, they are equal, unless they are
        // terms in which case they cannot be compared
        if self == other {
            if self.is_opaque() {
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
            (Self::Any, _) => None,
            (_, Self::Any) => None,
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
            (Self::Map, _) => Some(Less),
            (_, Self::Map) => Some(Greater),
            (Self::Nil, _) => Some(Less),
            (_, Self::Nil) => Some(Greater),
            (Self::Cons, _) => Some(Less),
            (_, Self::Cons) => Some(Greater),
            (Self::List(_), _) => Some(Less),
            (_, Self::List(_)) => Some(Greater),
            (Self::MaybeImproperList, _) => Some(Less),
            (_, Self::MaybeImproperList) => Some(Greater),
            (Self::Bitstring, Self::Binary) => Some(Equal),
            (Self::Binary, Self::Bitstring) => Some(Equal),
            (Self::Bitstring, _) => Some(Less),
            (Self::Binary, _) => Some(Less),
        }
    }
}

/// This enumeration covers all the types representable in SSA IR.
///
/// In addition to Erlang terms, we also need to represent primitive types and certain
/// internal runtime types which are used with primop instructions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    // This type is used to indicate that the type of an instruction is dynamic or unable to be typed
    Unknown,
    // This type is used to indicate an instruction that produces no results, and thus has no type
    Invalid,
    // Primitive types are used for some instructions which are low-level and do not directly produce
    // values which are used as terms, or are castable to term (e.g. integers)
    Primitive(PrimitiveType),
    // Term types are associated with values which correspond to syntax-level operations and are expected
    // to be used with runtime BIFs (built-in functions)
    Term(TermType),
    // Represents a function type that is calling-convention agnostic
    Function(FunctionType),
    // This type is equivalent to Rust's Never/! type, i.e. it indicates that a function never returns
    NoReturn,
    // This type maps to ErlangException in liblumen_rt
    Exception,
    // This type maps to Trace in liblumen_rt
    ExceptionTrace,
    // This type maps to ReceiveContext in lumen_rt_tiny
    RecvContext,
    // This type maps to ReceiveState in lumen_rt_tiny
    RecvState,
    // This type maps to BinaryBuilder in liblumen_rt
    BinaryBuilder,
    // This type maps to a match context
    MatchContext,
}
impl Default for Type {
    fn default() -> Type {
        Self::Unknown
    }
}
impl Type {
    pub fn tuple(arity: usize) -> Type {
        let mut elements = Vec::with_capacity(arity);
        elements.resize(arity, TermType::Any);
        Self::Term(TermType::Tuple(Some(elements)))
    }

    pub fn is_primitive(&self) -> bool {
        match self {
            Self::Primitive(_) => true,
            _ => false,
        }
    }

    pub fn is_term(&self) -> bool {
        match self {
            Self::Term(_) => true,
            _ => false,
        }
    }

    pub fn is_function(&self) -> bool {
        match self {
            Self::Function(_) | Self::Term(TermType::Fun(_)) => true,
            _ => false,
        }
    }

    pub fn is_special(&self) -> bool {
        match self {
            Self::Primitive(_) | Self::Term(_) => false,
            _ => true,
        }
    }

    pub fn as_primitive(&self) -> Option<PrimitiveType> {
        match self {
            Self::Primitive(prim) => Some(prim.clone()),
            _ => None,
        }
    }

    pub fn as_term(&self) -> Option<TermType> {
        match self {
            Self::Term(ty) => Some(ty.clone()),
            _ => None,
        }
    }

    pub fn is_unknown(&self) -> bool {
        match self {
            Self::Unknown => true,
            _ => false,
        }
    }
}
impl fmt::Display for Type {
    /// Print this type for display using the provided module context
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Unknown => f.write_str("?"),
            Self::Invalid => f.write_str("invalid"),
            Self::Primitive(prim) => write!(f, "{}", &prim),
            Self::Term(ty) => write!(f, "{}", &ty),
            Self::Function(ty) => write!(f, "{}", &ty),
            Self::NoReturn => f.write_char('!'),
            Self::Exception => f.write_str("exception"),
            Self::ExceptionTrace => f.write_str("trace"),
            Self::RecvContext => f.write_str("recv_context"),
            Self::RecvState => f.write_str("recv_state"),
            Self::BinaryBuilder => f.write_str("binary_builder"),
            Self::MatchContext => f.write_str("match_context"),
        }
    }
}
